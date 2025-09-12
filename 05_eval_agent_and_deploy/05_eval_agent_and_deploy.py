# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Evaluate and deploy a tool-calling LangGraph agent
# MAGIC
# MAGIC In this notebook you learn to:
# MAGIC
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about evaluating an agent using MLflow, see Databricks documentation ([Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-app)).
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "backoff",
    "langgraph", 
    "databricks-langchain", 
    "databricks-agents", 
    "mlflow-skinny", 
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

# MAGIC %md
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC #### Agent tools
# MAGIC This agent code adds the built-in Unity Catalog function `system.ai.python_exec` to the agent. The agent code also includes commented-out sample code for adding a vector search index to perform unstructured data retrieval.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))
# MAGIC
# MAGIC #### Wrap the LangGraph agent using the `ResponsesAgent` interface
# MAGIC
# MAGIC For compatibility with Databricks AI features, the `LangGraphResponsesAgent` class implements the `ResponsesAgent` interface to wrap the LangGraph agent.
# MAGIC
# MAGIC Databricks recommends using `ResponsesAgent` as it simplifies authoring multi-turn conversational agents using an open source standard. See MLflow's [ResponsesAgent documentation](https://www.mlflow.org/docs/latest/llms/responses-agent-intro/).
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output and tool-calling abilities. Since this notebook called `mlflow.langchain.autolog()`, you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

import shutil

# Clone the agent.py file to the current folder
shutil.copy("../04_create_agent_with_vsi_and_tools/agent.py", "./agent.py")

# COMMAND ----------

from agent import AGENT

result = AGENT.predict({"input": [{"role": "user", "content": "What color options are available for the Aria Modern Bookshelf?"}]})
print(result.model_dump(exclude_none=True))

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
# MAGIC
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).
# MAGIC
# MAGIC

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-langchain",
            f"langgraph=={get_distribution('langgraph').version}",
            f"backoff=={get_distribution('backoff').version}",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with Agent Evaluation
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics.
# MAGIC See Databricks documentation ([AWS]((https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-app)).
# MAGIC
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/custom-judge/)).

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines, Correctness
import mlflow.genai

# Define custom scorers tailored to product information evaluation
scorers = [
    #RetrievalGroundedness(),  # Pre-defined judge that checks against retrieval results
    Correctness(),
    RelevanceToQuery(),  # Checks if answer is relevant to the question
    Safety(),  # Checks for harmful or inappropriate content
    # Guidelines(
    #     guidelines="""Response must be clear and direct:
    #     - Answers the exact question asked
    #     - Uses lists for options, steps for instructions
    #     - No marketing fluff or extra background
    #     - Does not tell user to contact customer support
    #     - Concise but complete.""",
    #     name="clarity_and_structure",
    # ),
    #Guidelines(
    #    guidelines="""Response must include ALL expected facts:
    #    - Lists ALL colors/sizes if relevant (not partial lists)
    #    - States EXACT specs if relevant (e.g., "5 ATM" not "water resistant")
    #    - Includes ALL cleaning steps if asked
    #    Fails if ANY fact is missing or wrong.""",
    #    name="completeness_and_accuracy",
    #)
]

# COMMAND ----------

# create evaluation dataset
import pandas as pd

data = {
    "request": [
        "What color options are available for the Aria Modern Bookshelf?",
        "How should I clean the Aurora Oak Coffee Table to avoid damaging it?",
        "What sizes are available for the StormShield Pro Men's Weatherproof Jacket?"
    ],
    "expected_facts": [
        [
            "The Aria Modern Bookshelf is available in natural oak finish",
            "The Aria Modern Bookshelf is available in black finish",
            "The Aria Modern Bookshelf is available in white finish"
        ],
        [
            "Use a soft, slightly damp cloth for cleaning.",
            "Avoid using abrasive cleaners."
        ],
        [
            "The available sizes for the StormShield Pro Men's Weatherproof Jacket are Small, Medium, Large, XL, and XXL."
        ]
    ]
}

eval_dataset = pd.DataFrame(data)

# COMMAND ----------

# format it to match the format expected by mlflow.genai.evaluate
eval_data = []
for request, facts in zip(data["request"], data["expected_facts"]):
    expected = "\n".join(facts) if isinstance(facts, (list, tuple)) else str(facts)
    eval_data.append({
        "inputs": {
            "input": [
                {"role": "user", "content": request}
            ]
        },
        "expected_response": expected
    })
eval_data[0]

# COMMAND ----------

# run evaluation
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety

eval_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=scorers,  # add more scorers here if they're applicable
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "What color options are available for the Aria Modern Bookshelf?"}]},
    env_manager="uv",
)

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
model_name = "single_agent"
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
# can took up to 20 mins, will generate a feedback to collect feedback
agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"RemoveAfter": f"2026-01-01"},
)