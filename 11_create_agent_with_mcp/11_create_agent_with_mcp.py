# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy an MCP tool-calling LangGraph agent
# MAGIC
# MAGIC This notebook shows how to author a LangGraph agent that connects to MCP servers hosted on Databricks. You can choose between a Databricks-managed MCP server, a custom MCP server hosted as a Databricks app, or both simultaneously. To learn more about these options, see [MCP on Databricks](https://docs.databricks.com/aws/en/generative-ai/mcp/).
# MAGIC
# MAGIC
# MAGIC This notebook uses the [`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) interface for compatibility with Mosaic AI features. In this notebook you learn to:
# MAGIC
# MAGIC - Author a LangGraph agent (wrapped with `ResponsesAgent`) that calls MCP tools
# MAGIC - Manually test the agent
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U -qqqq mcp>=1.9 databricks-sdk[openai] databricks-agents>=1.0.0 databricks-mcp databricks-langchain uv langgraph==0.3.4
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the agent code
# MAGIC
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC The following cell creates a flexible, tool-using agent that integrates Databricks MCP servers with the Mosaic AI Agent Framework. Here’s what happens, at a high level:
# MAGIC
# MAGIC 1. **MCP tool wrappers**  
# MAGIC    Custom wrapper classes are defined so LangChain tools can interact with Databricks MCP servers. You can connect to Databricks-managed MCP servers, custom MCP servers hosted as a Databricks App, or both.
# MAGIC
# MAGIC 2. **Tool discovery & registration**  
# MAGIC    The agent automatically discovers available tools from the specified MCP server(s), turns their schemas into Python objects, and prepares them for the LLM.
# MAGIC
# MAGIC 3. **Define the LangGraph agent logic**  
# MAGIC    Define an agent workflow that does the following:
# MAGIC    - The agent reads messages (conversation/history).
# MAGIC    - If a tool (function) call is requested, it’s executed using the correct MCP tool.
# MAGIC    - The agent can loop, performing multiple tool calls as needed, until a final answer is ready.
# MAGIC
# MAGIC 4. **Wrap the LangGraph agent using the `ResponsesAgent` class**  
# MAGIC    The agent is wrapped using `ResponsesAgent` so it's compatible with the Mosaic AI.
# MAGIC    
# MAGIC 5. **MLflow autotracing**
# MAGIC    Enable MLflow autologging to start automatic tracing.

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import asyncio
# MAGIC import mlflow
# MAGIC import os
# MAGIC import json
# MAGIC from uuid import uuid4
# MAGIC from pydantic import BaseModel, create_model
# MAGIC from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union
# MAGIC
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC )
# MAGIC from databricks_mcp import DatabricksOAuthClientProvider, DatabricksMCPClient
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC     convert_to_openai_messages,
# MAGIC )
# MAGIC from langchain_core.tools import BaseTool, tool
# MAGIC
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC
# MAGIC from mcp import ClientSession
# MAGIC from mcp.client.streamable_http import streamablehttp_client as connect
# MAGIC
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC import nest_asyncio
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC ############################################
# MAGIC ## Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC system_prompt = """You are a customer service representative. Customers will come to you about questions around returning their order. You will prompt them for information to lookup their user name, and then lookup their orders to confirm their purchase for their product. You will look at their account information and the return policy to determine if the return is valid. Please be helpful and friendly, but don't violate the policy. You also have access to an inventory expert to help you see if certain items are in stock. This may be useful for exchanges. You also have access to a troubleshooting expert that can help you suggest fixes when customers are having issues with their products."""
# MAGIC
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Configure MCP Servers for your agent
# MAGIC ## This section sets up server connections so your agent can retrieve data or take actions.
# MAGIC ###############################################################################
# MAGIC
# MAGIC # TODO: Choose your MCP server connection type.
# MAGIC
# MAGIC # ----- Simple: Managed MCP Server (no extra setup required) -----
# MAGIC # Uses your Databricks Workspace settings and Personal Access Token (PAT) auth.
# MAGIC workspace_client = WorkspaceClient()
# MAGIC
# MAGIC # Managed MCP Servers: Ready to use with default settings above
# MAGIC host = workspace_client.config.host
# MAGIC MANAGED_MCP_SERVER_URLS = [
# MAGIC     f"{host}/api/2.0/mcp/genie/01f083cab80e1b6b8a23b98b61d8975c",
# MAGIC     f"{host}/api/2.0/mcp/vector-search/databricks_workshop/jywu",
# MAGIC     f"{host}/api/2.0/mcp/functions/databricks_workshop/jywu",
# MAGIC ]
# MAGIC
# MAGIC # ----- Advanced (optional): Custom MCP Server with OAuth -----
# MAGIC # For Databricks Apps hosting custom MCP servers, OAuth with a service principal is required.
# MAGIC # Uncomment and fill in your settings ONLY if connecting to a custom MCP server.
# MAGIC #
# MAGIC # import os
# MAGIC # workspace_client = WorkspaceClient(
# MAGIC #     host="<DATABRICKS_WORKSPACE_URL>",
# MAGIC #     client_id=os.getenv("DATABRICKS_CLIENT_ID"),
# MAGIC #     client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
# MAGIC #     auth_type="oauth-m2m",   # Enables machine-to-machine OAuth
# MAGIC # )
# MAGIC
# MAGIC # Custom MCP Servers: Add URLs below if needed (requires custom setup and OAuth above)
# MAGIC CUSTOM_MCP_SERVER_URLS = [
# MAGIC     # Example: "https://<custom-mcp-url>/mcp"
# MAGIC ]
# MAGIC
# MAGIC #####################
# MAGIC ## MCP Tool Creation
# MAGIC #####################
# MAGIC
# MAGIC # Define a custom LangChain tool that wraps functionality for calling MCP servers
# MAGIC class MCPTool(BaseTool):
# MAGIC     """Custom LangChain tool that wraps MCP server functionality"""
# MAGIC
# MAGIC     def __init__(self, name: str, description: str, args_schema: type, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
# MAGIC         # Initialize the tool
# MAGIC         super().__init__(
# MAGIC             name=name,
# MAGIC             description=description,
# MAGIC             args_schema=args_schema
# MAGIC         )
# MAGIC         # Store custom attributes: MCP server URL, Databricks workspace client, and whether the tool is for a custom server
# MAGIC         object.__setattr__(self, 'server_url', server_url)
# MAGIC         object.__setattr__(self, 'workspace_client', ws)
# MAGIC         object.__setattr__(self, 'is_custom', is_custom)
# MAGIC
# MAGIC     def _run(self, **kwargs) -> str:
# MAGIC         """Execute the MCP tool"""
# MAGIC         if self.is_custom:
# MAGIC             # Use the async method for custom MCP servers (OAuth required)
# MAGIC             return asyncio.run(self._run_custom_async(**kwargs))
# MAGIC         else:
# MAGIC             # Use managed MCP server via synchronous call
# MAGIC             mcp_client = DatabricksMCPClient(server_url=self.server_url, workspace_client=self.workspace_client)
# MAGIC             response = mcp_client.call_tool(self.name, kwargs)
# MAGIC             return "".join([c.text for c in response.content])
# MAGIC
# MAGIC     async def _run_custom_async(self, **kwargs) -> str:
# MAGIC         """Execute custom MCP tool asynchronously"""        
# MAGIC         async with connect(self.server_url, auth=DatabricksOAuthClientProvider(self.workspace_client)) as (
# MAGIC             read_stream,
# MAGIC             write_stream,
# MAGIC             _,
# MAGIC         ):
# MAGIC             # Create an async session with the server and call the tool
# MAGIC             async with ClientSession(read_stream, write_stream) as session:
# MAGIC                 await session.initialize()
# MAGIC                 response = await session.call_tool(self.name, kwargs)
# MAGIC                 return "".join([c.text for c in response.content])
# MAGIC
# MAGIC # Retrieve tool definitions from a custom MCP server (OAuth required)
# MAGIC async def get_custom_mcp_tools(ws: WorkspaceClient, server_url: str):
# MAGIC     """Get tools from a custom MCP server using OAuth"""    
# MAGIC     async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
# MAGIC         read_stream,
# MAGIC         write_stream,
# MAGIC         _,
# MAGIC     ):
# MAGIC         async with ClientSession(read_stream, write_stream) as session:
# MAGIC             await session.initialize()
# MAGIC             tools_response = await session.list_tools()
# MAGIC             return tools_response.tools
# MAGIC
# MAGIC # Retrieve tool definitions from a managed MCP server
# MAGIC def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
# MAGIC     """Get tools from a managed MCP server"""
# MAGIC     mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
# MAGIC     return mcp_client.list_tools()
# MAGIC
# MAGIC # Convert an MCP tool definition into a LangChain-compatible tool
# MAGIC def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
# MAGIC     """Create a LangChain tool from an MCP tool definition"""
# MAGIC     schema = mcp_tool.inputSchema.copy()
# MAGIC     properties = schema.get("properties", {})
# MAGIC     required = schema.get("required", [])
# MAGIC
# MAGIC     # Map JSON schema types to Python types for input validation
# MAGIC     TYPE_MAPPING = {
# MAGIC         "integer": int,
# MAGIC         "number": float,
# MAGIC         "boolean": bool
# MAGIC     }
# MAGIC     field_definitions = {}
# MAGIC     for field_name, field_info in properties.items():
# MAGIC         field_type_str = field_info.get("type", "string")
# MAGIC         field_type = TYPE_MAPPING.get(field_type_str, str)
# MAGIC
# MAGIC         if field_name in required:
# MAGIC             field_definitions[field_name] = (field_type, ...)
# MAGIC         else:
# MAGIC             field_definitions[field_name] = (field_type, None)
# MAGIC
# MAGIC     # Dynamically create a Pydantic schema for the tool's input arguments
# MAGIC     args_schema = create_model(
# MAGIC         f"{mcp_tool.name}Args",
# MAGIC         **field_definitions
# MAGIC     )
# MAGIC
# MAGIC     # Return a configured MCPTool instance
# MAGIC     return MCPTool(
# MAGIC         name=mcp_tool.name,
# MAGIC         description=mcp_tool.description or f"Tool: {mcp_tool.name}",
# MAGIC         args_schema=args_schema,
# MAGIC         server_url=server_url,
# MAGIC         ws=ws,
# MAGIC         is_custom=is_custom
# MAGIC     )
# MAGIC
# MAGIC # Gather all tools from managed and custom MCP servers into a single list
# MAGIC async def create_mcp_tools(ws: WorkspaceClient, 
# MAGIC                           managed_server_urls: List[str] = None, 
# MAGIC                           custom_server_urls: List[str] = None) -> List[MCPTool]:
# MAGIC     """Create LangChain tools from both managed and custom MCP servers"""
# MAGIC     tools = []
# MAGIC
# MAGIC     if managed_server_urls:
# MAGIC         # Load managed MCP tools
# MAGIC         for server_url in managed_server_urls:
# MAGIC             try:
# MAGIC                 mcp_tools = get_managed_mcp_tools(ws, server_url)
# MAGIC                 for mcp_tool in mcp_tools:
# MAGIC                     tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=False)
# MAGIC                     tools.append(tool)
# MAGIC             except Exception as e:
# MAGIC                 print(f"Error loading tools from managed server {server_url}: {e}")
# MAGIC
# MAGIC     if custom_server_urls:
# MAGIC         # Load custom MCP tools (async)
# MAGIC         for server_url in custom_server_urls:
# MAGIC             try:
# MAGIC                 mcp_tools = await get_custom_mcp_tools(ws, server_url)
# MAGIC                 for mcp_tool in mcp_tools:
# MAGIC                     tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=True)
# MAGIC                     tools.append(tool)
# MAGIC             except Exception as e:
# MAGIC                 print(f"Error loading tools from custom server {server_url}: {e}")
# MAGIC
# MAGIC     return tools
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC # The state for the agent workflow, including the conversation and any custom data
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC # Define the LangGraph agent that can call tools
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ):
# MAGIC     model = model.bind_tools(tools)  # Bind tools to the model
# MAGIC
# MAGIC     # Function to check if agent should continue or finish based on last message
# MAGIC     def should_continue(state: AgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If function (tool) calls are present, continue; otherwise, end
# MAGIC         if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     # Preprocess: optionally prepend a system prompt to the conversation history
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC
# MAGIC     model_runnable = preprocessor | model  # Chain the preprocessor and the model
# MAGIC
# MAGIC     # The function to invoke the model within the workflow
# MAGIC     def call_model(
# MAGIC         state: AgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(AgentState)  # Create the agent's state machine
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))  # Agent node (LLM)
# MAGIC     workflow.add_node("tools", ToolNode(tools))             # Tools node
# MAGIC
# MAGIC     workflow.set_entry_point("agent")  # Start at agent node
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",  # If the model requests a tool call, move to tools node
# MAGIC             "end": END,           # Otherwise, end the workflow
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")  # After tools are called, return to agent node
# MAGIC
# MAGIC     # Compile and return the tool-calling agent workflow
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC # ResponsesAgent class to wrap the compiled agent and make it compatible with Mosaic AI Responses API
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     # Convert a Responses-style message to a ChatCompletion format
# MAGIC     def _responses_to_cc(
# MAGIC         self, message: dict[str, Any]
# MAGIC     ) -> list[dict[str, Any]]:
# MAGIC         """Convert from a Responses API output item to ChatCompletion messages."""
# MAGIC         msg_type = message.get("type")
# MAGIC         if msg_type == "function_call":
# MAGIC             # Format tool/function call messages
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "assistant",
# MAGIC                     "content": "tool call",
# MAGIC                     "tool_calls": [
# MAGIC                         {
# MAGIC                             "id": message["call_id"],
# MAGIC                             "type": "function",
# MAGIC                             "function": {
# MAGIC                                 "arguments": message["arguments"],
# MAGIC                                 "name": message["name"],
# MAGIC                             },
# MAGIC                         }
# MAGIC                     ],
# MAGIC                 }
# MAGIC             ]
# MAGIC         elif msg_type == "message" and isinstance(message["content"], list):
# MAGIC             # Format regular content messages
# MAGIC             return [
# MAGIC                 {"role": message["role"], "content": content["text"]}
# MAGIC                 for content in message["content"]
# MAGIC             ]
# MAGIC         elif msg_type == "reasoning":
# MAGIC             # Reasoning steps as assistant messages
# MAGIC             return [{"role": "assistant", "content": json.dumps(message["summary"])}]
# MAGIC         elif msg_type == "function_call_output":
# MAGIC             # Function/tool outputs
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "tool",
# MAGIC                     "content": message["output"],
# MAGIC                     "tool_call_id": message["call_id"],
# MAGIC                 }
# MAGIC             ]
# MAGIC         # Pass through only the known, compatible fields
# MAGIC         compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
# MAGIC         filtered = {k: v for k, v in message.items() if k in compatible_keys}
# MAGIC         return [filtered] if filtered else []
# MAGIC
# MAGIC     # Convert a LangChain message to a Responses-format dictionary
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         """Convert from LangChain messages to Responses output item dictionaries"""
# MAGIC         for message in messages:
# MAGIC             message = message.model_dump()  # Convert the message model to dict
# MAGIC             role = message["type"]
# MAGIC             if role == "ai":
# MAGIC                 if tool_calls := message.get("tool_calls"):
# MAGIC                     # Return function call items for all tool calls present
# MAGIC                     return [
# MAGIC                         self.create_function_call_item(
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                             call_id=tool_call["id"],
# MAGIC                             name=tool_call["name"],
# MAGIC                             arguments=json.dumps(tool_call["args"]),
# MAGIC                         )
# MAGIC                         for tool_call in tool_calls
# MAGIC                     ]
# MAGIC                 else:
# MAGIC                     # Regular AI text message
# MAGIC                     return [
# MAGIC                         self.create_text_output_item(
# MAGIC                             text=message["content"],
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                         )
# MAGIC                     ]
# MAGIC             elif role == "tool":
# MAGIC                 # Output from tool/function execution
# MAGIC                 return [
# MAGIC                     self.create_function_call_output_item(
# MAGIC                         call_id=message["tool_call_id"],
# MAGIC                         output=message["content"],
# MAGIC                     )
# MAGIC                 ]
# MAGIC             elif role == "user":
# MAGIC                 # User messages as-is
# MAGIC                 return [message]
# MAGIC
# MAGIC     # Make a prediction (single-step) for the agent
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done" or event.type == "error"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     # Stream predictions for the agent, yielding output as it's generated
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = []
# MAGIC         for msg in request.input:
# MAGIC             cc_msgs.extend(self._responses_to_cc(msg.model_dump()))
# MAGIC
# MAGIC         # Stream events from the agent graph
# MAGIC         for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
# MAGIC             if event[0] == "updates":
# MAGIC                 # Stream updated messages from the workflow nodes
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if "messages" in node_data:
# MAGIC                         for item in self._langchain_to_responses(node_data["messages"]):
# MAGIC                             yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
# MAGIC             elif event[0] == "messages":
# MAGIC                 # Stream generated text message chunks
# MAGIC                 try:
# MAGIC                     chunk = event[1][0]
# MAGIC                     if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(delta=content, item_id=chunk.id),
# MAGIC                         )
# MAGIC                 except:
# MAGIC                     pass
# MAGIC
# MAGIC # Initialize the entire agent, including MCP tools and workflow
# MAGIC def initialize_agent():
# MAGIC     """Initialize the agent with MCP tools"""
# MAGIC     # Create MCP tools from the configured servers
# MAGIC     mcp_tools = asyncio.run(create_mcp_tools(
# MAGIC         ws=workspace_client,
# MAGIC         managed_server_urls=MANAGED_MCP_SERVER_URLS,
# MAGIC         custom_server_urls=CUSTOM_MCP_SERVER_URLS
# MAGIC     ))
# MAGIC
# MAGIC     # Create the agent graph with an LLM, tool set, and system prompt (if given)
# MAGIC     agent = create_tool_calling_agent(llm, mcp_tools, system_prompt)
# MAGIC     return LangGraphResponsesAgent(agent)
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = initialize_agent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output and tool-calling abilities. Since this notebook called `mlflow.langchain.autolog()`, you can view the trace for each step the agent takes.

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "What tools do you have?"}]})

# COMMAND ----------

from agent import AGENT
AGENT.predict({"input": [{"role": "user", "content": "What color options are available for the Aria Modern Bookshelf??"}]})

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What tools do you have?"}]}
):
    print(chunk, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [Deploy an agent that connects to Databricks MCP servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp#deploy-your-agent).

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from pkg_resources import get_distribution
from mlflow.models.resources import (
  DatabricksVectorSearchIndex,
  DatabricksServingEndpoint,
  DatabricksSQLWarehouse,
  DatabricksFunction,
  DatabricksGenieSpace,
  DatabricksTable,
  DatabricksUCConnection,
  DatabricksApp
)

CATALOG_NAME = "databricks_workshop"
SCHEMA_NAME = "jywu"

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME), 
    DatabricksFunction(function_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.get_latest_return"),
    DatabricksFunction(function_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.get_order_history"),
    DatabricksFunction(function_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.get_return_policy"),
    DatabricksFunction(function_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.get_todays_date"),
    DatabricksVectorSearchIndex(index_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.product_docs_index"),
    DatabricksTable(table_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.customer_services"),
    DatabricksTable(table_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.policies"),
    DatabricksTable(table_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.inventories"),
    DatabricksGenieSpace(genie_space_id="01f083cab80e1b6b8a23b98b61d8975c"),
    DatabricksSQLWarehouse(warehouse_id="f33f0c83be7369d5")
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        pip_requirements=[
            "databricks-mcp",
            f"mlflow=={get_distribution('mlflow').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mcp=={get_distribution('mcp').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://docs.databricks.com/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://docs.databricks.com/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness, Correctness

eval_dataset = [
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "Do you have SoundWave Pro X5 in stock?"
                }
            ]
        },
        "expected_response": "Yes, we have 24 in stock."
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety(), Correctness()], # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Do you have SoundWave Pro X5 in stock?"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
model_name = "single_agent_with_mcp"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME, 
    uc_registered_model_info.version,
    tags = {"RemoveAfter": "2026-01-01"}
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)).
