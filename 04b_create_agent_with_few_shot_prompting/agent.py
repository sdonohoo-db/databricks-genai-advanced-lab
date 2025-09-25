import json
import csv
import pandas as pd
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, List, Dict
from uuid import uuid4

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

############################################
# Define your LLM endpoint and configuration
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
CATALOG_NAME = "databricks_workshop"
SCHEMA_NAME = "jywu"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

###############################################################################
## Few-shot prompting configuration
## Define input/output templates and examples for few-shot learning
###############################################################################

# System prompt for processing CSV files with Jira stories
SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that generates test cases from Jira stories. You must return ONLY valid JSON output with no additional text, commentary, or explanations.

CRITICAL: Your response must contain ONLY the JSON structure shown in the output template. Do not include any introductory text, explanatory comments, or step-by-step descriptions.

You will receive stories and templates. Generate test cases following the output template format exactly.
Each test case must use the same test_case_id as the original story's issue_id.

Return ONLY the JSON structure - nothing else."""

def read_csv_file(file_name: str) -> pd.DataFrame:
    """Read CSV file from current directory"""
    return pd.read_csv(file_name)

def read_story_files(csv_files: List[str]) -> List[Dict]:
    """Read story files from CSV files in current directory"""
    all_stories = []

    for csv_file in csv_files:
        try:
            print(f"Reading CSV file: {csv_file}")

            # Read the CSV file
            df = read_csv_file(csv_file)

            # Standardize column names (handle different naming conventions)
            df.columns = df.columns.str.lower().str.strip()

            # Map common column variations
            column_mapping = {
                'id': 'issue_id',
                'story_id': 'issue_id',
                'ticket_id': 'issue_id',
                'user_story': 'description',
                'story': 'description',
                'desc': 'description',
                'criteria': 'acceptance_criteria',
                'acceptance': 'acceptance_criteria',
                'ac': 'acceptance_criteria'
            }

            df = df.rename(columns=column_mapping)

            # Validate required columns
            required_columns = ['issue_id', 'description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing required columns {missing_columns} in {csv_file}")
                continue

            # Convert each row to a story object
            for _, row in df.iterrows():
                # Handle acceptance criteria - could be string, list, or missing
                acceptance_criteria = []
                if 'acceptance_criteria' in row and pd.notna(row['acceptance_criteria']):
                    criteria_value = row['acceptance_criteria']
                    if isinstance(criteria_value, str):
                        # Try different delimiters
                        if '; ' in criteria_value:
                            acceptance_criteria = criteria_value.split('; ')
                        elif ';' in criteria_value:
                            acceptance_criteria = criteria_value.split(';')
                        elif ', ' in criteria_value:
                            acceptance_criteria = criteria_value.split(', ')
                        elif ',' in criteria_value:
                            acceptance_criteria = criteria_value.split(',')
                        elif '\n' in criteria_value:
                            acceptance_criteria = criteria_value.split('\n')
                        else:
                            acceptance_criteria = [criteria_value]
                        # Clean up criteria
                        acceptance_criteria = [c.strip() for c in acceptance_criteria if c.strip()]
                    elif isinstance(criteria_value, list):
                        acceptance_criteria = criteria_value

                story = {
                    "issue_id": str(row['issue_id']).strip(),
                    "description": str(row['description']).strip(),
                    "acceptance_criteria": acceptance_criteria
                }
                all_stories.append(story)

            print(f"Successfully read {len(df)} stories from {csv_file}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    print(f"Total stories loaded: {len(all_stories)}")
    return all_stories

def clean_json_response(content: str) -> str:
    """Extract only the JSON structure from LLM response, removing any extra text"""
    try:
        # Remove common prefixes and suffixes that LLMs might add
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]

        # Find the JSON structure - look for the first { and last }
        start_idx = content.find('{')
        if start_idx == -1:
            return content  # No JSON found, return as-is

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return content  # No proper JSON structure found

        json_content = content[start_idx:end_idx+1]

        # Validate that it's valid JSON
        json.loads(json_content)
        return json_content

    except (json.JSONDecodeError, Exception):
        # If cleaning fails, return original content
        return content

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_few_shot_agent(
    model: LanguageModelLike,
    system_prompt_template: Optional[str] = None,
):
    """Create a few-shot learning agent that processes requests with embedded templates."""

    if system_prompt_template is None:
        system_prompt_template = SYSTEM_PROMPT_TEMPLATE

    system_prompt = system_prompt_template

    def call_model(state: AgentState, config: RunnableConfig):
        # Get the user message from state
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {"messages": [AIMessage(content="Please provide a request with CSV files and templates.")]}

        try:
            # Parse the user request JSON
            user_content = user_messages[-1].content
            request_data = json.loads(user_content)

            # Extract components from the request
            csv_files = request_data.get("csv_files", [])
            input_template = request_data.get("input_template")
            output_template = request_data.get("output_template")

            if not input_template or not output_template:
                return {"messages": [AIMessage(content="Both input_template and output_template must be provided in the request.")]}

            # Read all files and convert to stories
            all_stories = read_story_files(csv_files)

            if not all_stories:
                return {"messages": [AIMessage(content="No stories found in the provided files.")]}

            # Create the complete request for the LLM
            llm_request = {
                "stories": all_stories,
                "input_template": input_template,
                "output_template": output_template
            }

            # Create the user message for the LLM with actual story data
            llm_user_message = f"""Stories:
{json.dumps(all_stories, indent=2)}

Output template:
{json.dumps(output_template, indent=2)}

Generate test cases for ALL stories following the exact output template format. Each test_case_id must match the story's issue_id. Return ONLY the JSON structure - no other text."""

            # Create the full conversation with system prompt
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=llm_user_message)
            ]

            # Call the model
            response = model.invoke(messages, config)

            # Clean the response to ensure only JSON is returned
            cleaned_content = clean_json_response(response.content)

            # Create a new AIMessage with cleaned content
            cleaned_response = AIMessage(content=cleaned_content, id=response.id if hasattr(response, 'id') else None)
            return {"messages": [cleaned_response]}

        except json.JSONDecodeError:
            return {"messages": [AIMessage(content="Invalid JSON format in request. Please provide a valid JSON object with csv_files, input_template, and output_template.")]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}

    # Create the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        compatible_keys = ["role", "content", "name"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses output item dictionaries"""
        results = []
        for message in messages:
            if isinstance(message, AIMessage):
                results.append(
                    self.create_text_output_item(
                        text=message.content,
                        id=getattr(message, 'id', str(uuid4())),
                    )
                )
            elif isinstance(message, HumanMessage):
                results.append({
                    "role": "user",
                    "content": message.content
                })
        return results

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Convert request to messages
        messages = []
        for msg in request.input:
            msg_dict = msg.model_dump()
            if msg_dict.get("role") == "user":
                messages.append(HumanMessage(content=msg_dict["content"]))

        # Run the agent
        for event in self.agent.stream(
            {"messages": messages},
            stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if "messages" in node_data:
                        for item in self._langchain_to_responses(node_data["messages"]):
                            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                try:
                    chunk = event[1][0] if event[1] else None
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(
                                delta=chunk.content,
                                item_id=getattr(chunk, 'id', str(uuid4()))
                            ),
                        )
                except Exception as e:
                    print(f"Streaming error: {e}")


# Create the agent object
mlflow.langchain.autolog()
agent = create_few_shot_agent(llm)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)