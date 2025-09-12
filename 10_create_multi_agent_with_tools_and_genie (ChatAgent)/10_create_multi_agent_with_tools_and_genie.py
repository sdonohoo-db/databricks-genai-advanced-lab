# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy a multi-agent system with Genie
# MAGIC
# MAGIC This notebook demonstrates how to build a multi-agent system using Mosaic AI Agent Framework and [LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), where [Genie](https://www.databricks.com/product/ai-bi/genie) is one of the agents.
# MAGIC In this notebook, you:
# MAGIC 1. Author a multi-agent system using LangGraph.
# MAGIC 1. Wrap the LangGraph agent with MLflow `ChatAgent` to ensure compatibility with Databricks features.
# MAGIC 1. Manually test the multi-agent system's output.
# MAGIC 1. Log and deploy the multi-agent system.
# MAGIC
# MAGIC This example is based on [LangGraph documentation - Multi-agent supervisor example](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.md)
# MAGIC
# MAGIC ## Why use a Genie agent?
# MAGIC
# MAGIC Multi-agent systems consist of multiple AI agents working together, each with specialized capabilities. As one of those agents, Genie allows users to interact with their structured data using natural language.
# MAGIC
# MAGIC Unlike SQL functions which can only run pre-defined queries, Genie has the flexibility to create novel queries to answer user questions.

# COMMAND ----------

# MAGIC %pip install -U -qqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "mlflow-skinny",
    "langgraph",
    "databricks-langchain",
    "databricks-agents",
    "uv"
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

# MAGIC %run ../00_setup/config

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# COMMAND ----------

help(GenieAgent)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define the multi-agent system
# MAGIC
# MAGIC Create a multi-agent system in LangGraph using a supervisor agent node directing the following agent nodes:
# MAGIC - **GenieAgent**: The Genie agent that queries and reasons over structured data.
# MAGIC - **Tool-calling agent**: An agent that calls Unity Catalog function tools.
# MAGIC
# MAGIC In this example, the tool-calling agent uses the built-in Unity Catalog function `system.ai.python_exec` to execute Python code.
# MAGIC For examples of other tools you can add to your agents, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool)).
# MAGIC
# MAGIC
# MAGIC #### Wrap the LangGraph agent using the `ChatAgent` interface
# MAGIC
# MAGIC Databricks recommends using `ChatAgent` to ensure compatibility with Databricks AI features and to simplify authoring multi-turn conversational agents using an open source standard. 
# MAGIC
# MAGIC The `LangGraphChatAgent` class implements the `ChatAgent` interface to wrap the LangGraph agent.
# MAGIC
# MAGIC See MLflow's [ChatAgent documentation](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent).
# MAGIC
# MAGIC #### Write agent code to file
# MAGIC
# MAGIC Define the agent code in a single cell below. This lets you write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import functools
# MAGIC import os
# MAGIC from typing import Any, Generator, Literal, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_langchain import (
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     DatabricksFunctionClient,
# MAGIC     set_uc_function_client
# MAGIC )
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client) 
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt import create_react_agent
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC from pydantic import BaseModel
# MAGIC
# MAGIC ###################################################
# MAGIC ## Create a GenieAgent with access to a Genie Space
# MAGIC ###################################################
# MAGIC
# MAGIC # TODO add GENIE_SPACE_ID and a description for this space
# MAGIC # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
# MAGIC # Example description: This Genie agent can answer questions based on a database containing tables related to enterprise software sales, including accounts, opportunities, opportunity history, fiscal periods, quotas, targets, teams, and users. Use Genie to fetch and analyze data from these tables by specifying the relevant columns and filters. Genie can execute SQL queries to provide precise data insights based on your questions.
# MAGIC GENIE_SPACE_ID = "01f083cab80e1b6b8a23b98b61d8975c"
# MAGIC genie_agent_description = (
# MAGIC     "Use Genie ONLY for database/analytics on transactional inventory facts (e.g., stock counts, inventory movements, sales over time). NOT for product specs."
# MAGIC )
# MAGIC genie_agent = GenieAgent(
# MAGIC     genie_space_id=GENIE_SPACE_ID,
# MAGIC     genie_agent_name="Genie",
# MAGIC     description=genie_agent_description,
# MAGIC     # client=WorkspaceClient(
# MAGIC     #     host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
# MAGIC     #     token=os.getenv("DATABRICKS_GENIE_PAT"),
# MAGIC     # ),
# MAGIC )
# MAGIC
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC # multi-agent Genie works best with claude 3.7 or gpt 4o models.
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC # LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC
# MAGIC ############################################################
# MAGIC # Create a code agent
# MAGIC # You can also create agents with access to additional tools
# MAGIC ############################################################
# MAGIC tools = []
# MAGIC
# MAGIC # TODO if desired, add additional tools and update the description of this agent
# MAGIC # uc_tool_names = ["system.ai.*"]
# MAGIC # uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC # tools.extend(uc_toolkit.tools)
# MAGIC # code_agent_description = (
# MAGIC #     "The Coder agent specializes in solving programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts.",
# MAGIC # )
# MAGIC # code_agent = create_react_agent(llm, tools=tools)
# MAGIC
# MAGIC # add uc_tools to the agent
# MAGIC uc_tool_names = ["databricks_workshop.jywu.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # add vector search indexes to the agent
# MAGIC vs_tool = VectorSearchRetrieverTool(
# MAGIC   index_name="databricks_workshop.jywu.product_docs_index",
# MAGIC   tool_name="product_docs_retriever",
# MAGIC   tool_description="Retrieves information about various products including introduction, how to use it, trouble shooting, etc."
# MAGIC )
# MAGIC tools.extend([vs_tool])
# MAGIC service_agent_description = (
# MAGIC     "Use this agent for product information/specs (colors, sizes, materials, instructions, troubleshooting) AND for return/exchange policies. For any product-info question, ALWAYS call the tool 'product_docs_retriever' before answering."
# MAGIC )
# MAGIC service_agent = create_react_agent(llm, tools=tools)
# MAGIC
# MAGIC #############################
# MAGIC # Define the supervisor agent
# MAGIC #############################
# MAGIC
# MAGIC # TODO update the max number of iterations between supervisor and worker nodes
# MAGIC # before returning to the user
# MAGIC MAX_ITERATIONS = 10
# MAGIC
# MAGIC worker_descriptions = {
# MAGIC     "Genie": genie_agent_description,
# MAGIC     "Service": service_agent_description,
# MAGIC }
# MAGIC
# MAGIC formatted_descriptions = "\n".join(
# MAGIC     f"- {name}: {desc}" for name, desc in worker_descriptions.items()
# MAGIC )
# MAGIC
# MAGIC system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
# MAGIC options = ["FINISH"] + list(worker_descriptions.keys())
# MAGIC FINISH = {"next_node": "FINISH"}
# MAGIC
# MAGIC def supervisor_agent(state):
# MAGIC     count = state.get("iteration_count", 0) + 1
# MAGIC     if count > MAX_ITERATIONS:
# MAGIC         return FINISH
# MAGIC     
# MAGIC     class nextNode(BaseModel):
# MAGIC         next_node: Literal[tuple(options)]
# MAGIC
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC     )
# MAGIC     supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
# MAGIC     next_node = supervisor_chain.invoke(state).next_node
# MAGIC     
# MAGIC     # if routed back to the same node, exit the loop
# MAGIC     if state.get("next_node") == next_node:
# MAGIC         return FINISH
# MAGIC     return {
# MAGIC         "iteration_count": count,
# MAGIC         "next_node": next_node
# MAGIC     }
# MAGIC
# MAGIC #######################################
# MAGIC # Define our multiagent graph structure
# MAGIC #######################################
# MAGIC
# MAGIC
# MAGIC def agent_node(state, agent, name):
# MAGIC     result = agent.invoke(state)
# MAGIC     return {
# MAGIC         "messages": [
# MAGIC             {
# MAGIC                 "role": "assistant",
# MAGIC                 "content": result["messages"][-1].content,
# MAGIC                 "name": name,
# MAGIC             }
# MAGIC         ]
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC def final_answer(state):
# MAGIC     prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages."
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: state["messages"] + [{"role": "system", "content": prompt}]
# MAGIC     )
# MAGIC     final_answer_chain = preprocessor | llm
# MAGIC     return {"messages": [final_answer_chain.invoke(state)]}
# MAGIC
# MAGIC
# MAGIC class AgentState(ChatAgentState):
# MAGIC     next_node: str
# MAGIC     iteration_count: int
# MAGIC
# MAGIC
# MAGIC service_node = functools.partial(agent_node, agent=service_agent, name="Service")
# MAGIC genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")
# MAGIC
# MAGIC workflow = StateGraph(AgentState)
# MAGIC workflow.add_node("Service", service_node)
# MAGIC workflow.add_node("Genie", genie_node)
# MAGIC workflow.add_node("supervisor", supervisor_agent)
# MAGIC workflow.add_node("final_answer", final_answer)
# MAGIC
# MAGIC workflow.set_entry_point("supervisor")
# MAGIC # We want our workers to ALWAYS "report back" to the supervisor when done
# MAGIC for worker in worker_descriptions.keys():
# MAGIC     workflow.add_edge(worker, "supervisor")
# MAGIC
# MAGIC # Let the supervisor decide which next node to go
# MAGIC workflow.add_conditional_edges(
# MAGIC     "supervisor",
# MAGIC     lambda x: x["next_node"],
# MAGIC     {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
# MAGIC )
# MAGIC workflow.add_edge("final_answer", END)
# MAGIC multi_agent = workflow.compile()
# MAGIC
# MAGIC ###################################
# MAGIC # Wrap our multi-agent in ChatAgent
# MAGIC ###################################
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {
# MAGIC             "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
# MAGIC         }
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {
# MAGIC             "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
# MAGIC         }
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg})
# MAGIC                     for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = LangGraphChatAgent(multi_agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC **TODO**: Replace this placeholder `input_example` with a domain-specific prompt for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# test the GenieAgent
from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "What is the return policy?"}]})

# COMMAND ----------

# test the agent with vector search indexes
from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "What color options are available for the Aria Modern Bookshelf?"}]})

# COMMAND ----------

from agent import AGENT

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Explain the datasets and capabilities that the Genie agent has access to.",
        }
    ]
}
for event in AGENT.predict_stream(input_example):
  print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/log-agent#resources) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).
# MAGIC
# MAGIC   - **TODO**: If the SQL Warehouse powering your Genie space has secured permissions, include the warehouse ID and table name in your resources to enable passthrough authentication. ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/log-agent#-specify-resources-for-automatic-authentication-passthrough-system-authentication) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#-specify-resources-for-automatic-authentication-passthrough-system-authentication)).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import GENIE_SPACE_ID, LLM_ENDPOINT_NAME, tools
from databricks_langchain import UnityCatalogTool, VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from pkg_resources import get_distribution

CATALOG_NAME = "databricks_workshop"
SCHEMA_NAME = "jywu"

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
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
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    DatabricksSQLWarehouse(warehouse_id="f33f0c83be7369d5")
]
# for tool in tools:
#     if isinstance(tool, VectorSearchRetrieverTool):
#         resources.extend(tool.resources)
#     elif isinstance(tool, UnityCatalogTool):
#         resources.append(DatabricksFunction(function_name=tool.uc_function_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data=input_example,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
model_name = "multi_agent"
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
    tags={"RemoveAfter": "2026-01-01"},
)
