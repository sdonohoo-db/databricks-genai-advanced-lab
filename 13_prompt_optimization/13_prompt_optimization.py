# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Optimization Tutorial: Multi-Step Agent with Dataset Creation
# MAGIC This notebook demonstrates an end-to-end flow to improve your prompt for a multi-step agent using the `mlflow.genai.optimize_prompt` API. In this notebook you learn to:
# MAGIC - How to collect traces for LLM calls during a multi-step agent execution
# MAGIC - How to create evaluation dataset from MLflow traces
# MAGIC - How to run prompt optimization with your prompt, evaluation metrics and dataset
# MAGIC
# MAGIC [Databricks [doc](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/prompt-version-mgmt/prompt-registry/automatically-optimize-prompts) | MLflow [doc](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Agent
# MAGIC The first step is defining the AI agent. In this notebook, we use LangGraph to define an agent to extract the main topic of an article. The agent consists of two LLM calls. The first LLM call summarize the long document content into a short summary and the second call extracts the main topic from the summary.

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow>=3.1.0 langchain-community langchain-openai beautifulsoup4 langgraph dspy databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "mlflow",
    "langchain-community",
    "langchain-openai",
    "beautifulsoup4",
    "langgraph",
    "dspy",
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

import os
import mlflow
from mlflow.entities import Prompt
mlflow.set_registry_uri("databricks-uc")
# os.environ["OPENAI_API_KEY"] = <Your API key>

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

split_docs = text_splitter.split_documents(docs)
print(f"Generated {len(split_docs)} documents.")

# COMMAND ----------

# from langchain.chat_models import init_chat_model

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be available. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-3-70b-instruct" 

)

# COMMAND ----------

# First prompt for summarization.
summary_prompt = mlflow.genai.register_prompt(name=f"{catalog_name}.{schema_name}.summary_prompt", template="Write a concise summary of the following:{{content}}")

# COMMAND ----------

print(summary_prompt)

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

summary_chain = llm | StrOutputParser()

@mlflow.trace()
def call_summary_chain(content):
  return summary_chain.invoke([HumanMessage(summary_prompt.format(content=content))])

# COMMAND ----------

# Second prompt for topic extraction.
topic_prompt = mlflow.genai.register_prompt(name=f"{catalog_name}.{schema_name}.topic_prompt",
                       template="""
The following is the summary:
{{summary}}
Extract the main topic in a few words.
Return the response in JSON format: {"topic": "..."}
""")

topic_chain = llm | JsonOutputParser()

@mlflow.trace()
def call_topic_chain(summary):
  return topic_chain.invoke([HumanMessage(topic_prompt.format(summary=summary))])

# COMMAND ----------

print(topic_prompt)

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage

@mlflow.trace
def agent(content):
  summary = call_summary_chain(content=content)
  return call_topic_chain(summary=summary)["topic"]

# COMMAND ----------

# Enable Autologging
mlflow.langchain.autolog()

# COMMAND ----------

# Run the agent
for doc in split_docs:
  try:
    print(agent(doc.page_content))
  except Exception as e:
    print(e)
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Creation
# MAGIC
# MAGIC Create an evaluation dataset from the generated traces using the `mlflow.genai.datasets` API. In this example, we focus on the second LLM call for topic extraction.

# COMMAND ----------

import mlflow

# Extract the inputs and outputs of the second LLM call
traces = mlflow.search_traces(extract_fields=[
  "call_topic_chain.inputs",
  "call_topic_chain.outputs",
])

# COMMAND ----------

traces.head(10)

# COMMAND ----------

from mlflow.genai import datasets

EVAL_DATASET_NAME=f"{catalog_name}.{schema_name}.data_for_prompt_optimization"
dataset = datasets.create_dataset(EVAL_DATASET_NAME)

# COMMAND ----------

# Create a dataset by treating the agent outputs as the default expectations.
traces = traces.rename(
    columns={
      "call_topic_chain.inputs": "inputs",
      "call_topic_chain.outputs": "expectations",
    }
)[["inputs", "expectations"]]
traces = traces.dropna()
dataset.merge_records(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labeling
# MAGIC Currently, the expectation tab of the dataset contains the agent outputs. To run the prompt optimization, it's essential to have a good quality label. Go to "evaluation" tab -> "dataset" tab and modify the expectations.
# MAGIC After you finish the labeling, run the following command to download the eval dataset.

# COMMAND ----------

dataset = datasets.get_dataset(EVAL_DATASET_NAME)
dataset.merge_records([])

# COMMAND ----------

dataset = dataset.to_df()
dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize
# MAGIC Finally, let's run `mlflow.genai.optimize_prompt` and optimize your prompt. In the code below, we use the built-in Correctness scorer as our objective function. The optimized prompt is automatically stored in the Prompt registry. Check the new prompt template after running the optimizer.

# COMMAND ----------

import os
from typing import Any
import mlflow
from mlflow.genai.scorers import Correctness
from mlflow.genai.optimize import OptimizerConfig, LLMParams
from mlflow.genai.scorers import scorer

_correctness = Correctness()

@scorer
def correctness(inputs, outputs, expectations):
    expectations = { "expected_response": expectations.get("topic") }
    return _correctness(inputs=inputs, outputs=outputs, expectations=expectations).value == "yes"

# Optimize the prompt
result = mlflow.genai.optimize_prompt(
    target_llm_params=LLMParams(model_name=f"databricks/databricks-claude-3-7-sonnet"), #target_llm_params=LLMParams(model_name="openai/gpt-4.1-mini"),
    prompt=topic_prompt,
    train_data=dataset,
    scorers=[correctness],
    optimizer_config=OptimizerConfig(
        num_instruction_candidates=8,
        max_few_show_examples=2,
        verbose=False, # turn it on to see the full logs
    )
)

# The optimized prompt is automatically registered as a new version
# Open the prompt registry web site to check the new prompt
print(f"The new prompt URI: {result.prompt.uri}")
