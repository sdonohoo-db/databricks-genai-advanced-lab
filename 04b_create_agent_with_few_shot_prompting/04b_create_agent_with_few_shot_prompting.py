# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author a few-shot prompting agent
# MAGIC
# MAGIC This notebook shows how to author an agent using few-shot prompting with templatized examples instead of tool-calling. The agent uses input/output example pairs to learn how to transform inputs into the desired output format. In this notebook you learn to:
# MAGIC
# MAGIC - Author a few-shot prompting agent wrapped with `ResponsesAgent`
# MAGIC - Configure the agent with input/output template examples
# MAGIC - Manually test the agent's output with different example patterns
# MAGIC
# MAGIC This approach is useful when you want to show the agent specific input/output patterns rather than giving it access to external tools or data retrieval capabilities.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain langgraph==0.5.3 databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "langgraph",
    "databricks-langchain",
    "databricks-agents",
    "mlflow-skinny"
]

# Get installed packages and their versions
installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}

# Check and print the version of specified packages
for package_name in packages_to_check:
    version = installed_packages.get(package_name)
    if version:
        print(f"{package_name}: {version}")
    else:
        print(f"{package_name} is not installed.")

# COMMAND ----------

import mlflow

mlflow_version = mlflow.__version__
display(mlflow_version)

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This agent uses few-shot prompting with templatized input/output examples instead of external tools.
# MAGIC
# MAGIC #### Few-shot prompting approach
# MAGIC This agent accepts:
# MAGIC - `input_template`: Example input format to show the agent what kind of input to expect
# MAGIC - `output_template`: Example output format to show the agent how to structure responses
# MAGIC - `examples`: List of input/output example pairs for few-shot learning
# MAGIC
# MAGIC #### Wrap the LangGraph agent using the `ResponsesAgent` interface
# MAGIC
# MAGIC For compatibility with Databricks AI features, the `LangGraphResponsesAgent` class implements the `ResponsesAgent` interface to wrap the LangGraph agent.

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC import csv
# MAGIC import pandas as pd
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, List, Dict
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC     HumanMessage,
# MAGIC     SystemMessage,
# MAGIC )
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and configuration
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC CATALOG_NAME = "databricks_workshop"
# MAGIC SCHEMA_NAME = "jywu"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Few-shot prompting configuration
# MAGIC ## Define input/output templates and examples for few-shot learning
# MAGIC ###############################################################################
# MAGIC
# MAGIC # System prompt for processing CSV files with Jira stories
# MAGIC SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that generates test cases from Jira stories. You must return ONLY valid JSON output with no additional text, commentary, or explanations.
# MAGIC
# MAGIC CRITICAL: Your response must contain ONLY the JSON structure shown in the output template. Do not include any introductory text, explanatory comments, or step-by-step descriptions.
# MAGIC
# MAGIC You will receive stories and templates. Generate test cases following the output template format exactly.
# MAGIC Each test case must use the same test_case_id as the original story's issue_id.
# MAGIC
# MAGIC Return ONLY the JSON structure - nothing else."""
# MAGIC
# MAGIC def read_csv_file(file_name: str) -> pd.DataFrame:
# MAGIC     """Read CSV file from current directory"""
# MAGIC     return pd.read_csv(file_name)
# MAGIC
# MAGIC def read_story_files(csv_files: List[str]) -> List[Dict]:
# MAGIC     """Read story files from CSV files in current directory"""
# MAGIC     all_stories = []
# MAGIC
# MAGIC     for csv_file in csv_files:
# MAGIC         try:
# MAGIC             print(f"Reading CSV file: {csv_file}")
# MAGIC
# MAGIC             # Read the CSV file
# MAGIC             df = read_csv_file(csv_file)
# MAGIC
# MAGIC             # Standardize column names (handle different naming conventions)
# MAGIC             df.columns = df.columns.str.lower().str.strip()
# MAGIC
# MAGIC             # Map common column variations
# MAGIC             column_mapping = {
# MAGIC                 'id': 'issue_id',
# MAGIC                 'story_id': 'issue_id',
# MAGIC                 'ticket_id': 'issue_id',
# MAGIC                 'user_story': 'description',
# MAGIC                 'story': 'description',
# MAGIC                 'desc': 'description',
# MAGIC                 'criteria': 'acceptance_criteria',
# MAGIC                 'acceptance': 'acceptance_criteria',
# MAGIC                 'ac': 'acceptance_criteria'
# MAGIC             }
# MAGIC
# MAGIC             df = df.rename(columns=column_mapping)
# MAGIC
# MAGIC             # Validate required columns
# MAGIC             required_columns = ['issue_id', 'description']
# MAGIC             missing_columns = [col for col in required_columns if col not in df.columns]
# MAGIC             if missing_columns:
# MAGIC                 print(f"Warning: Missing required columns {missing_columns} in {csv_file}")
# MAGIC                 continue
# MAGIC
# MAGIC             # Convert each row to a story object
# MAGIC             for _, row in df.iterrows():
# MAGIC                 # Handle acceptance criteria - could be string, list, or missing
# MAGIC                 acceptance_criteria = []
# MAGIC                 if 'acceptance_criteria' in row and pd.notna(row['acceptance_criteria']):
# MAGIC                     criteria_value = row['acceptance_criteria']
# MAGIC                     if isinstance(criteria_value, str):
# MAGIC                         # Try different delimiters
# MAGIC                         if '; ' in criteria_value:
# MAGIC                             acceptance_criteria = criteria_value.split('; ')
# MAGIC                         elif ';' in criteria_value:
# MAGIC                             acceptance_criteria = criteria_value.split(';')
# MAGIC                         elif ', ' in criteria_value:
# MAGIC                             acceptance_criteria = criteria_value.split(', ')
# MAGIC                         elif ',' in criteria_value:
# MAGIC                             acceptance_criteria = criteria_value.split(',')
# MAGIC                         elif '\n' in criteria_value:
# MAGIC                             acceptance_criteria = criteria_value.split('\n')
# MAGIC                         else:
# MAGIC                             acceptance_criteria = [criteria_value]
# MAGIC                         # Clean up criteria
# MAGIC                         acceptance_criteria = [c.strip() for c in acceptance_criteria if c.strip()]
# MAGIC                     elif isinstance(criteria_value, list):
# MAGIC                         acceptance_criteria = criteria_value
# MAGIC
# MAGIC                 story = {
# MAGIC                     "issue_id": str(row['issue_id']).strip(),
# MAGIC                     "description": str(row['description']).strip(),
# MAGIC                     "acceptance_criteria": acceptance_criteria
# MAGIC                 }
# MAGIC                 all_stories.append(story)
# MAGIC
# MAGIC             print(f"Successfully read {len(df)} stories from {csv_file}")
# MAGIC
# MAGIC         except Exception as e:
# MAGIC             print(f"Error processing {csv_file}: {e}")
# MAGIC             continue
# MAGIC
# MAGIC     print(f"Total stories loaded: {len(all_stories)}")
# MAGIC     return all_stories
# MAGIC
# MAGIC def clean_json_response(content: str) -> str:
# MAGIC     """Extract only the JSON structure from LLM response, removing any extra text"""
# MAGIC     try:
# MAGIC         # Remove common prefixes and suffixes that LLMs might add
# MAGIC         content = content.strip()
# MAGIC
# MAGIC         # Remove markdown code blocks if present
# MAGIC         if content.startswith('```json'):
# MAGIC             content = content[7:]
# MAGIC         if content.startswith('```'):
# MAGIC             content = content[3:]
# MAGIC         if content.endswith('```'):
# MAGIC             content = content[:-3]
# MAGIC
# MAGIC         # Find the JSON structure - look for the first { and last }
# MAGIC         start_idx = content.find('{')
# MAGIC         if start_idx == -1:
# MAGIC             return content  # No JSON found, return as-is
# MAGIC
# MAGIC         # Find the matching closing brace
# MAGIC         brace_count = 0
# MAGIC         end_idx = -1
# MAGIC         for i in range(start_idx, len(content)):
# MAGIC             if content[i] == '{':
# MAGIC                 brace_count += 1
# MAGIC             elif content[i] == '}':
# MAGIC                 brace_count -= 1
# MAGIC                 if brace_count == 0:
# MAGIC                     end_idx = i
# MAGIC                     break
# MAGIC
# MAGIC         if end_idx == -1:
# MAGIC             return content  # No proper JSON structure found
# MAGIC
# MAGIC         json_content = content[start_idx:end_idx+1]
# MAGIC
# MAGIC         # Validate that it's valid JSON
# MAGIC         json.loads(json_content)
# MAGIC         return json_content
# MAGIC
# MAGIC     except (json.JSONDecodeError, Exception):
# MAGIC         # If cleaning fails, return original content
# MAGIC         return content
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC
# MAGIC
# MAGIC def create_few_shot_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     system_prompt_template: Optional[str] = None,
# MAGIC ):
# MAGIC     """Create a few-shot learning agent that processes requests with embedded templates."""
# MAGIC
# MAGIC     if system_prompt_template is None:
# MAGIC         system_prompt_template = SYSTEM_PROMPT_TEMPLATE
# MAGIC
# MAGIC     system_prompt = system_prompt_template
# MAGIC
# MAGIC     def call_model(state: AgentState, config: RunnableConfig):
# MAGIC         # Get the user message from state
# MAGIC         user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
# MAGIC         if not user_messages:
# MAGIC             return {"messages": [AIMessage(content="Please provide a request with CSV files and templates.")]}
# MAGIC
# MAGIC         try:
# MAGIC             # Parse the user request JSON
# MAGIC             user_content = user_messages[-1].content
# MAGIC             request_data = json.loads(user_content)
# MAGIC
# MAGIC             # Extract components from the request
# MAGIC             csv_files = request_data.get("csv_files", [])
# MAGIC             input_template = request_data.get("input_template")
# MAGIC             output_template = request_data.get("output_template")
# MAGIC
# MAGIC             if not input_template or not output_template:
# MAGIC                 return {"messages": [AIMessage(content="Both input_template and output_template must be provided in the request.")]}
# MAGIC
# MAGIC             # Read all files and convert to stories
# MAGIC             all_stories = read_story_files(csv_files)
# MAGIC
# MAGIC             if not all_stories:
# MAGIC                 return {"messages": [AIMessage(content="No stories found in the provided files.")]}
# MAGIC
# MAGIC             # Create the user message for the LLM with actual story data
# MAGIC             llm_user_message = f"""Stories:
# MAGIC {json.dumps(all_stories, indent=2)}
# MAGIC
# MAGIC Output template:
# MAGIC {json.dumps(output_template, indent=2)}
# MAGIC
# MAGIC Generate test cases for ALL stories following the exact output template format. Each test_case_id must match the story's issue_id. Return ONLY the JSON structure - no other text."""
# MAGIC
# MAGIC             # Create the full conversation with system prompt
# MAGIC             messages = [
# MAGIC                 SystemMessage(content=system_prompt),
# MAGIC                 HumanMessage(content=llm_user_message)
# MAGIC             ]
# MAGIC
# MAGIC             # Call the model
# MAGIC             response = model.invoke(messages, config)
# MAGIC
# MAGIC             # Clean the response to ensure only JSON is returned
# MAGIC             cleaned_content = clean_json_response(response.content)
# MAGIC
# MAGIC             # Create a new AIMessage with cleaned content
# MAGIC             cleaned_response = AIMessage(content=cleaned_content, id=response.id if hasattr(response, 'id') else None)
# MAGIC             return {"messages": [cleaned_response]}
# MAGIC
# MAGIC         except json.JSONDecodeError:
# MAGIC             return {"messages": [AIMessage(content="Invalid JSON format in request. Please provide a valid JSON object with csv_files, input_template, and output_template.")]}
# MAGIC         except Exception as e:
# MAGIC             return {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}
# MAGIC
# MAGIC     # Create the workflow
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_edge("agent", END)
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
# MAGIC         """Convert from a Responses API output item to ChatCompletion messages."""
# MAGIC         compatible_keys = ["role", "content", "name"]
# MAGIC         filtered = {k: v for k, v in message.items() if k in compatible_keys}
# MAGIC         return [filtered] if filtered else []
# MAGIC
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         """Convert from LangChain messages to Responses output item dictionaries"""
# MAGIC         results = []
# MAGIC         for message in messages:
# MAGIC             if isinstance(message, AIMessage):
# MAGIC                 results.append(
# MAGIC                     self.create_text_output_item(
# MAGIC                         text=message.content,
# MAGIC                         id=getattr(message, 'id', str(uuid4())),
# MAGIC                     )
# MAGIC                 )
# MAGIC             elif isinstance(message, HumanMessage):
# MAGIC                 results.append({
# MAGIC                     "role": "user",
# MAGIC                     "content": message.content
# MAGIC                 })
# MAGIC         return results
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         # Convert request to messages
# MAGIC         messages = []
# MAGIC         for msg in request.input:
# MAGIC             msg_dict = msg.model_dump()
# MAGIC             if msg_dict.get("role") == "user":
# MAGIC                 messages.append(HumanMessage(content=msg_dict["content"]))
# MAGIC
# MAGIC         # Run the agent
# MAGIC         for event in self.agent.stream(
# MAGIC             {"messages": messages},
# MAGIC             stream_mode=["updates", "messages"]
# MAGIC         ):
# MAGIC             if event[0] == "updates":
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if "messages" in node_data:
# MAGIC                         for item in self._langchain_to_responses(node_data["messages"]):
# MAGIC                             yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
# MAGIC             elif event[0] == "messages":
# MAGIC                 try:
# MAGIC                     chunk = event[1][0] if event[1] else None
# MAGIC                     if isinstance(chunk, AIMessageChunk) and chunk.content:
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(
# MAGIC                                 delta=chunk.content,
# MAGIC                                 item_id=getattr(chunk, 'id', str(uuid4()))
# MAGIC                             ),
# MAGIC                         )
# MAGIC                 except Exception as e:
# MAGIC                     print(f"Streaming error: {e}")
# MAGIC
# MAGIC
# MAGIC # Create the agent object
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_few_shot_agent(llm)
# MAGIC AGENT = LangGraphResponsesAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent with CSV file processing
# MAGIC
# MAGIC Test the agent with synthetic CSV files containing Jira stories to see how it processes multiple files and generates test cases.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT
import json


# MAGIC %md
# MAGIC ## Test Authentication Stories with Security-focused Schema
# MAGIC Each story type should use a different output schema that matches its domain.

# COMMAND ----------

# Test Authentication stories with security-focused test schema
auth_request = {
    "csv_files": [
        "stories_authentication.csv"
    ],
    "input_template": {
        "issue_id": "AUTH-123",
        "description": "As a user, I want to authenticate securely so that my account is protected.",
        "acceptance_criteria": [
            "User credentials are validated",
            "Security measures are enforced",
            "Access is granted upon successful authentication"
        ]
    },
    "output_template": {
        "security_test_cases": [
            {
                "test_id": "AUTH-123",
                "security_category": "Authentication",
                "test_scenario": "Valid login with correct credentials",
                "preconditions": ["User has valid account", "System is accessible"],
                "test_steps": [
                    "Navigate to login page",
                    "Enter valid username and password",
                    "Click login button"
                ],
                "expected_outcome": "User successfully authenticated and redirected to dashboard",
                "security_assertions": [
                    "Credentials are encrypted in transit",
                    "Session token is generated",
                    "No sensitive data exposed in response"
                ]
            },
            {
                "test_id": "AUTH-123",
                "security_category": "Authentication",
                "test_scenario": "Invalid credentials handling",
                "preconditions": ["System is accessible"],
                "test_steps": [
                    "Navigate to login page",
                    "Enter invalid username/password",
                    "Click login button"
                ],
                "expected_outcome": "Authentication fails with appropriate error message",
                "security_assertions": [
                    "No sensitive information leaked in error",
                    "Account not locked after single failure",
                    "Error message is generic"
                ]
            }
        ]
    }
}

auth_json = json.dumps(auth_request, indent=2)

print("Processing authentication stories only:")
auth_result = AGENT.predict({"input": [{"role": "user", "content": auth_json}]})
print("Authentication processing complete!")

# Print the JSON result directly
if hasattr(auth_result, 'output') and auth_result.output:
    for output_item in auth_result.output:
        if isinstance(output_item, dict) and 'content' in output_item:
            print("\n=== Authentication Test Cases (JSON) ===")
            print(output_item['content'])
        else:
            print("\n=== Raw Output ===")
            print(output_item)
else:
    print("No output found in result")

# COMMAND ----------

# Pretty print the authentication results for better readability
print("=== PRETTY-PRINTED AUTHENTICATION RESULTS ===")
if hasattr(auth_result, 'output') and auth_result.output:
    for i, output_item in enumerate(auth_result.output):
        if isinstance(output_item, dict) and 'content' in output_item:
            try:
                # Parse and pretty print the JSON
                parsed_json = json.loads(output_item['content'])
                print(f"\n--- Output Item {i+1} (Pretty Formatted) ---")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"\n--- Output Item {i+1} (Raw Text) ---")
                print(output_item['content'])
        else:
            print(f"\n--- Output Item {i+1} (Non-dict) ---")
            print(json.dumps(output_item, indent=2, ensure_ascii=False))
else:
    print("No output to pretty print")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test E-commerce Stories with BDD-style Schema
# MAGIC E-commerce tests use Given-When-Then format for better user journey testing.

# COMMAND ----------

# Test E-commerce stories with BDD (Behavior-Driven Development) schema
ecommerce_request = {
    "csv_files": [
        "stories_ecommerce.csv"
    ],
    "input_template": {
        "issue_id": "SHOP-401",
        "description": "As a customer, I want to browse products easily so that I can find what I'm looking for",
        "acceptance_criteria": [
            "Product catalog is accessible",
            "Search functionality works",
            "Filters can be applied"
        ]
    },
    "output_template": {
        "bdd_test_scenarios": [
            {
                "story_id": "SHOP-401",
                "feature": "Product Discovery",
                "scenario_name": "Browse products by category",
                "priority": "High",
                "given": ["User is on the main shopping page", "Product categories are visible"],
                "when": ["User clicks on 'Electronics' category"],
                "then": [
                    "All electronics products are displayed",
                    "Product count is shown",
                    "Filter options are available"
                ],
                "acceptance_criteria_covered": ["Product catalog is accessible"]
            },
            {
                "story_id": "SHOP-401",
                "feature": "Product Search",
                "scenario_name": "Search for specific products",
                "priority": "High",
                "given": ["User is on any page", "Search box is visible"],
                "when": ["User enters 'laptop' in search box", "User clicks search button"],
                "then": [
                    "Relevant laptop products are displayed",
                    "Search results are sorted by relevance",
                    "No irrelevant products shown"
                ],
                "acceptance_criteria_covered": ["Search functionality works", "Filters can be applied"]
            }
        ]
    }
}

ecommerce_json = json.dumps(ecommerce_request, indent=2)

print("Processing e-commerce stories with BDD schema:")
ecommerce_result = AGENT.predict({"input": [{"role": "user", "content": ecommerce_json}]})
print("E-commerce BDD processing complete!")

# Print the JSON result directly
if hasattr(ecommerce_result, 'output') and ecommerce_result.output:
    for output_item in ecommerce_result.output:
        if isinstance(output_item, dict) and 'content' in output_item:
            print("\n=== E-commerce BDD Test Cases (JSON) ===")
            print(output_item['content'])
        else:
            print("\n=== Raw Output ===")
            print(output_item)
else:
    print("No output found in result")

# COMMAND ----------

# Pretty print the e-commerce results for better readability
print("=== PRETTY-PRINTED E-COMMERCE BDD RESULTS ===")
if hasattr(ecommerce_result, 'output') and ecommerce_result.output:
    for i, output_item in enumerate(ecommerce_result.output):
        if isinstance(output_item, dict) and 'content' in output_item:
            try:
                # Parse and pretty print the JSON
                parsed_json = json.loads(output_item['content'])
                print(f"\n--- Output Item {i+1} (Pretty Formatted) ---")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"\n--- Output Item {i+1} (Raw Text) ---")
                print(output_item['content'])
        else:
            print(f"\n--- Output Item {i+1} (Non-dict) ---")
            print(json.dumps(output_item, indent=2, ensure_ascii=False))
else:
    print("No output to pretty print")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Profile Stories with UX-focused Schema
# MAGIC Profile/user management tests focus on user experience and usability validation.

# COMMAND ----------

# Test Profile stories with UX (User Experience) focused schema
profile_request = {
    "csv_files": [
        "stories_profile.csv"
    ],
    "input_template": {
        "issue_id": "PROF-123",
        "description": "As a user, I want to manage my profile settings so that my information is up to date",
        "acceptance_criteria": [
            "Profile information can be edited",
            "Changes are saved successfully",
            "Updated information is displayed"
        ]
    },
    "output_template": {
        "ux_test_cases": [
            {
                "story_id": "PROF-123",
                "ux_category": "Profile Management",
                "user_journey": "Update personal information",
                "personas": ["Regular User", "New User"],
                "devices_tested": ["Desktop", "Mobile"],
                "user_actions": [
                    {
                        "step": "Navigate to profile settings",
                        "expected_ui": "Profile icon is visible and clickable",
                        "usability_check": "Navigation is intuitive"
                    },
                    {
                        "step": "Edit name field",
                        "expected_ui": "Field is clearly editable with validation",
                        "usability_check": "Clear feedback on field requirements"
                    },
                    {
                        "step": "Save changes",
                        "expected_ui": "Save button is prominently displayed",
                        "usability_check": "Success message is clear and confirmatory"
                    }
                ],
                "success_metrics": [
                    "User can complete task without assistance",
                    "Time to complete is under 2 minutes",
                    "No confusion about next steps"
                ],
                "accessibility_requirements": [
                    "Keyboard navigation supported",
                    "Screen reader compatible",
                    "High contrast mode functional"
                ]
            }
        ]
    }
}

profile_json = json.dumps(profile_request, indent=2)

print("Processing profile stories with UX schema:")
profile_result = AGENT.predict({"input": [{"role": "user", "content": profile_json}]})
print("Profile UX processing complete!")

# Print the JSON result directly
if hasattr(profile_result, 'output') and profile_result.output:
    for output_item in profile_result.output:
        if isinstance(output_item, dict) and 'content' in output_item:
            print("\n=== Profile UX Test Cases (JSON) ===")
            print(output_item['content'])
        else:
            print("\n=== Raw Output ===")
            print(output_item)
else:
    print("No output found in result")

# COMMAND ----------

# Pretty print the profile results for better readability
print("=== PRETTY-PRINTED PROFILE UX RESULTS ===")
if hasattr(profile_result, 'output') and profile_result.output:
    for i, output_item in enumerate(profile_result.output):
        if isinstance(output_item, dict) and 'content' in output_item:
            try:
                # Parse and pretty print the JSON
                parsed_json = json.loads(output_item['content'])
                print(f"\n--- Output Item {i+1} (Pretty Formatted) ---")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"\n--- Output Item {i+1} (Raw Text) ---")
                print(output_item['content'])
        else:
            print(f"\n--- Output Item {i+1} (Non-dict) ---")
            print(json.dumps(output_item, indent=2, ensure_ascii=False))
else:
    print("No output to pretty print")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Multiple CSV Files with Same Schema
# MAGIC Test processing multiple CSV files that should all use the same output schema.

# COMMAND ----------

# Test with multiple CSV files using the same schema (Security compliance focus)
multi_file_request = {
    "csv_files": [
        "stories_authentication.csv",
        "stories_profile.csv"
    ],
    "input_template": {
        "issue_id": "SEC-123",
        "description": "As a user, I want secure access to my account so that my personal data is protected.",
        "acceptance_criteria": [
            "User authentication is secure",
            "Personal data is protected",
            "Access controls are enforced"
        ]
    },
    "output_template": {
        "compliance_test_cases": [
            {
                "test_id": "SEC-123",
                "compliance_category": "Data Protection",
                "regulatory_framework": "GDPR",
                "test_objective": "Verify secure user authentication",
                "risk_level": "High",
                "test_procedures": [
                    {
                        "procedure": "Attempt login with valid credentials",
                        "security_validation": "Verify password encryption",
                        "compliance_check": "Ensure audit logging is active"
                    },
                    {
                        "procedure": "Verify session management",
                        "security_validation": "Check token expiration",
                        "compliance_check": "Validate session timeout controls"
                    }
                ],
                "expected_security_outcome": "User authenticated securely with full audit trail",
                "compliance_evidence": [
                    "Authentication logs captured",
                    "Encryption standards met",
                    "Access controls verified"
                ]
            }
        ]
    }
}

multi_file_json = json.dumps(multi_file_request, indent=2)

print("Processing multiple CSV files (Authentication + Profile) with same compliance schema:")
multi_file_result = AGENT.predict({"input": [{"role": "user", "content": multi_file_json}]})
print("Multi-file compliance processing complete!")

# Print the JSON result directly
if hasattr(multi_file_result, 'output') and multi_file_result.output:
    for output_item in multi_file_result.output:
        if isinstance(output_item, dict) and 'content' in output_item:
            print("\n=== Multi-file Compliance Test Cases (JSON) ===")
            print(output_item['content'])
        else:
            print("\n=== Raw Output ===")
            print(output_item)
else:
    print("No output found in result")

# COMMAND ----------

# Pretty print the multi-file results for better readability
print("=== PRETTY-PRINTED MULTI-FILE COMPLIANCE RESULTS ===")
if hasattr(multi_file_result, 'output') and multi_file_result.output:
    for i, output_item in enumerate(multi_file_result.output):
        if isinstance(output_item, dict) and 'content' in output_item:
            try:
                # Parse and pretty print the JSON
                parsed_json = json.loads(output_item['content'])
                print(f"\n--- Output Item {i+1} (Pretty Formatted) ---")
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"\n--- Output Item {i+1} (Raw Text) ---")
                print(output_item['content'])
        else:
            print(f"\n--- Output Item {i+1} (Non-dict) ---")
            print(json.dumps(output_item, indent=2, ensure_ascii=False))
else:
    print("No output to pretty print")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrates how to create a few-shot prompting agent that can:
# MAGIC
# MAGIC 1. **Process multiple CSV files** containing Jira stories in a single request
# MAGIC 2. **Generate test cases** for all stories across all files
# MAGIC 3. **Maintain ID consistency** - each test case uses the same ID as its source story
# MAGIC 4. **Adapt to different formats** using input/output templates
# MAGIC 5. **Support streaming responses** for real-time processing feedback
# MAGIC
# MAGIC The agent uses few-shot learning to understand the expected input/output patterns and can be dynamically reconfigured with new templates for different test case formats (e.g., Given-When-Then vs. Steps-Expected Result).