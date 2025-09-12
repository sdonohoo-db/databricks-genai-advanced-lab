# Databricks notebook source
# MAGIC %md
# MAGIC ## Collect domain expert feedback with the Review App UI
# MAGIC Check out more detail in [documentation](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces)

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.3.1" openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "mlflow",
    "openai"
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
# MAGIC ### Step 1: Define Experiment

# COMMAND ----------

import os
import mlflow 
from databricks.sdk import WorkspaceClient
# Let's re-use an existing experiment
mlflow.set_tracking_uri("databricks")
xp_name = os.getcwd().rsplit("/", 1)[0]+"/05_eval_agent_and_deploy/05_eval_agent_and_deploy"
mlflow.set_experiment(xp_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Define Labeling Schemas
# MAGIC Labeling schemas define the questions and input types that domain experts will use to provide feedback on your traces. You can use MLflow's built-in schemas or create custom ones tailored to your specific evaluation criteria.
# MAGIC
# MAGIC There are two main types of labeling schemas:
# MAGIC
# MAGIC - Expectation Type (```type="expectation"```): Used when the expert provides a "ground truth" or a correct answer. For example, providing the ```expected_facts``` for a RAG system's response. These labels can often be directly used in evaluation datasets.
# MAGIC - Feedback Type (```type="feedback"```): Used for subjective assessments, ratings, or classifications. For example, rating a response on a scale of 1-5 for politeness, or classifying if a response met certain criteria.
# MAGIC
# MAGIC See the [Labeling Schemas documentation](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-schemas) to understand the various input methods for your schemas, such as categorical choices (radio buttons), numeric scales, or free-form text.
# MAGIC
# MAGIC

# COMMAND ----------


from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText

# Collect feedback on the answer
answer_quality = create_label_schema(
    name="answer_quality",
    type="feedback",
    title="Is this answer concise and helpful?",
    input=InputCategorical(options=["Yes", "No"]),
    instruction="Please provide a rationale below.",
    enable_comment=True,
    overwrite=True,
)

# Collect a ground truth answer
expected_answer = create_label_schema(
    name="expected_answer",
    type="expectation",
    title="Please provide the correct answer for the user's request.",
    input=InputText(),
    overwrite=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Create a Labeling Session
# MAGIC A Labeling Session is a special type of MLflow Run organizes a set of traces for review by specific experts using selected labeling schemas. It acts as a queue for the review process.
# MAGIC
# MAGIC See the [Labeling Session documentation](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-sessions) for more details.
# MAGIC
# MAGIC Here's how to create a labeling session:

# COMMAND ----------

from mlflow.genai.labeling import create_labeling_session

# Create the Labeling Session with the schemas we created in the previous step
label_answers = create_labeling_session(
    name="sme_label_answers",
    assigned_users=[], # Leaving empty for now - give permissions in Step 5
    label_schemas=[answer_quality.name, expected_answer.name],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Generate traces and add to the Labeling Session
# MAGIC Once your labeling session is created, you need to add traces to it. Traces are copied into the labeling session, so any labels or modifications made during the review process do not affect your original logged traces.
# MAGIC
# MAGIC You can add any trace in your MLflow Experiment. See the [Labeling Session documentation](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-sessions) for more details.

# COMMAND ----------

import mlflow 
mlflow.search_runs(experiment_names=[xp_name])

# COMMAND ----------

# Get most recent eval run
runs = mlflow.search_runs(experiment_names=[xp_name], filter_string="status = 'FINISHED'", order_by=["start_time DESC"], max_results=1)

# Query for the traces we just generated from that run.
# You can also paste run_id here
traces = mlflow.search_traces(run_id=runs.run_id[0])
# traces = mlflow.search_traces(run_id="3f61a91d382a489c92dba7961988f6f0")

# Add the traces to the session
label_answers.add_traces(traces)

# Print the URL to share with your domain experts
print(f"Share this Review App with your team: {label_answers.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Share the Review App with Experts
# MAGIC Once your labeling session is populated with traces, you can share its URL with your domain experts. They can use this URL to access the Review App, view the traces assigned to them (or pick from unassigned ones), and provide feedback using the labeling schemas you configured.

# COMMAND ----------

# Assign the feedback to the session
label_answers.set_assigned_users(["himanshu.gupta@databricks.com"]) # TODO: Change the email address to your user(s) or group(s)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: View and Use Collected Labels
# MAGIC After your domain experts have completed their reviews, the collected feedback is attached to the traces within the labeling session. You can retrieve these labels programmatically to analyze them or use them to create evaluation datasets.
# MAGIC
# MAGIC Labels are stored as ```Assessment``` objects on each Trace within the Labeling Session.

# COMMAND ----------

# Updated requirement for a code sample showing:

# Goal: Demonstrate how to retrieve and process collected assessments (labels) from traces within a completed or in-progress labeling session.
# Outline:
# 1. Assume a 'labeling_session' object (from previous steps) is available.
# 2. Get the 'mlflow_run_id' from the 'labeling_session'
# 3. Use 'mlflow.search_traces(run_id=session_run_id)' to fetch all traces logged within that specific labeling session run.
# 4. Iterate through the retrieved traces (e.g., rows in the DataFrame returned by search_traces).
# 5. For each trace, access its 'assessments'. Assessments are stored as MLflow assessment objects (such as Feedback and Expectation types) in a list within the 'assessments' column of the trace.
#    - Access assessment attributes using dot notation: assessment.name (schema name), assessment.value (expert's input), assessment.rationale (comments), assessment.source.source_id (assessor identifier), and assessment.create_time_ms (timestamp).
# 6. Compile these assessments from all traces into a Pandas DataFrame.
#    - The DataFrame should have columns such as: 'trace_id', 'assessment_name', 'assessment_value', 'assessment_comment', 'assessor_id', 'timestamp'.
# 7. Print the head of the resulting DataFrame to display some of the collected labels.
# 8. Demonstrate how to filter this DataFrame, for example, to show only assessments related to a specific schema (e.g., 'summary_quality').

import mlflow
import pandas as pd

# Get the experiment ID from the labeling session object
experiment_id = label_answers.experiment_id
print(f"Session ID: {experiment_id}")

# Fetch all traces from the labeling session
traces_df = mlflow.search_traces(experiment_ids=[experiment_id])
print(f"Found {len(traces_df)} traces")

# Extract assessments from traces
assessments = []
for _, trace in traces_df.iterrows():
    if trace['assessments']:
        for assessment in trace['assessments']:
            if 'feedback' in assessment and pd.notna(assessment['feedback']):
                assessments.append({
                    'trace_id': trace['trace_id'],
                    'assessor_id': assessment['source']['source_id'],
                    'assessment_name': assessment['assessment_name'],
                    'assessment_value': assessment['feedback']['value'],
                    #'assessment_comment': assessment['rationale'],
                    'timestamp': assessment['create_time']
            })

# Create DataFrame with all assessments
assessments_df = pd.DataFrame(assessments)
print(f"\nCollected {len(assessments_df)} assessments")

if len(assessments_df) > 0:
    # Display the assessments
    print("\nAssessments Preview:")
    display(assessments_df.head())

    # Filter assessments by schema name
    summary_quality_assessments = assessments_df[
        assessments_df['assessment_name'] == 'summary_quality']

    print(f"\nSummary quality assessments: {len(summary_quality_assessments)} found")
    if not summary_quality_assessments.empty:
        print(summary_quality_assessments[['trace_id', 'assessment_value']].head())

# COMMAND ----------

traces_df['assessments'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step: Convert to Evaluation Datasets
# MAGIC (Similar to the Mlflow 3.0 Evaluation Notebook.)
# MAGIC
# MAGIC Labels of "expectation" type (e.g., ```expected_summary``` from our example) are particularly useful for creating [Evaluation Datasets](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset). These datasets can then be used with ```mlflow.genai.evaluate()``` to systematically test new versions of your GenAI application against expert-defined ground truth.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create a dataset

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
import time
from databricks.connect import DatabricksSession

# This table will be created in the above UC schema
UC_PREFIX = f"{catalog_name}.{schema_name}"
evaluation_dataset_table_name = f"{UC_PREFIX}.sme_eval"

# If the evaluation dataset already exists, remove the table
spark.sql(f"DROP TABLE IF EXISTS {evaluation_dataset_table_name}")

eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name=f"{evaluation_dataset_table_name}",
)
print(f"Created evaluation dataset: {evaluation_dataset_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Add records to your dataset
# MAGIC Follow [Approach 2: Create from domain expert labels](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset#approach-2-create-from-domain-expert-labels)
# MAGIC
# MAGIC Note: The documentation may not be updated depedning on when you are viewing this.

# COMMAND ----------

import mlflow.genai.labeling as labeling

# Get a labeling sessions
all_sessions = labeling.get_labeling_sessions()
print(f"Found {len(all_sessions)} sessions")

# Sync the first session
if len(all_sessions) > 0:
  all_sessions[0].sync(to_dataset=f"{evaluation_dataset_table_name}")
