# Databricks notebook source
# MAGIC %md
# MAGIC # Build Simple Tools
# MAGIC - **SQL Functions**: Create queries that access data critical to steps in the customer service workflow for processing a return.
# MAGIC - **Simple Python Function**: Create and register a Python function to overcome some common limitations of language models.

# COMMAND ----------

# MAGIC %pip install -U -qqqq unitycatalog-langchain[databricks]
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pkg_resources

# List of packages to check
packages_to_check = [
    "unitycatalog-langchain"
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

spark.sql(f"USE CATALOG `{catalog_name}`")
spark.sql(f"USE SCHEMA  `{schema_name}`")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT current_catalog(), current_schema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Service Return Processing Workflow
# MAGIC
# MAGIC Below is a structured outline of the **key steps** a customer service agent would typically follow when **processing a return**. This workflow ensures consistency and clarity across your support team.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Get the Latest Return in the Processing Queue
# MAGIC - **Action**: Identify and retrieve the most recent return request from the ticketing or returns system.  
# MAGIC - **Why**: Ensures you’re working on the most urgent or next-in-line customer issue.
# MAGIC
# MAGIC ---

# COMMAND ----------

spark.sql(f"""
          SELECT * FROM {catalog_name}.{schema_name}.customer_services LIMIT 5
          """).display()

# COMMAND ----------

# DBTITLE 1,Create a function registered to Unity Catalog
spark.sql(f"""
          CREATE OR REPLACE FUNCTION get_latest_return()
          RETURNS TABLE(purchase_date DATE, issue_category STRING, issue_description STRING, name STRING)
          COMMENT 'Returns the most recent customer service interaction, such as returns.'
          RETURN (
          SELECT
          CAST(date_time AS DATE) AS purchase_date,
          issue_category,
          issue_description,
          name
          FROM {catalog_name}.{schema_name}.customer_services
          ORDER BY date_time DESC
          LIMIT 1
          )
          """)

# COMMAND ----------

# DBTITLE 1,Test function call to retrieve latest return
# MAGIC %sql
# MAGIC SELECT * FROM get_latest_return()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. Use the User Name to Look Up the Order History
# MAGIC - **Action**: Query your order management system or customer database using the Username.  
# MAGIC - **Why**: Reviewing past purchases, return patterns, and any specific notes helps you determine appropriate next steps (e.g., confirm eligibility for return).
# MAGIC
# MAGIC ###### Note: We've doing a little trick to give the LLM extra context into the current date by adding todays_date in the function's response
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Create function that retrieves order history based on userID
spark.sql(f"""
          CREATE OR REPLACE FUNCTION get_order_history(user_name STRING)
          RETURNS TABLE (returns_last_12_months INT, issue_category STRING, todays_date DATE)
          COMMENT 'This takes the user_name of a customer as an input and returns the number of returns and the issue category'
          LANGUAGE SQL
          RETURN
          SELECT count(*) as returns_last_12_months, issue_category, now() as todays_date
          FROM {catalog_name}.{schema_name}.customer_services
          WHERE name = user_name
          GROUP BY issue_category
          """)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_order_history('Nicolas Pelaez')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. Retrieve Company Policies
# MAGIC - **Action**: Access the internal knowledge base or policy documents related to returns, refunds, and exchanges.  
# MAGIC - **Why**: Verifying you’re in compliance with company guidelines prevents potential errors and conflicts.
# MAGIC
# MAGIC ---

# COMMAND ----------

spark.sql(f"""
          SELECT * FROM {catalog_name}.{schema_name}.policies LIMIT 5
          """).display()

# COMMAND ----------

# DBTITLE 1,Create function to retrieve return policy
spark.sql(f"""
          CREATE OR REPLACE FUNCTION get_return_policy()
          RETURNS TABLE (
            policy           STRING,
            policy_details   STRING,
            last_updated     DATE
            )
            COMMENT 'Returns the details of the Return Policy'
            LANGUAGE SQL
            RETURN (
              SELECT
              policy,
              policy_details,
              last_updated
              FROM {catalog_name}.{schema_name}.policies
              WHERE policy = 'Return Policy'
              LIMIT 1
              )
          """)

# COMMAND ----------

# DBTITLE 1,Test function to retrieve return policy
# MAGIC %sql
# MAGIC SELECT * FROM get_return_policy()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. Python functions can be used as well! Here's an example
# MAGIC - **Action**: Provide a **Python function** that can supply the Large Language Model (LLM) with the current date.  
# MAGIC - **Why**: Automating date retrieval helps in scheduling pickups, refund timelines, and communication deadlines.
# MAGIC
# MAGIC ###### Note: For this lab we will not be using this function but leaving as example.
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Very simple Python function
def get_todays_date() -> str:
    """
    Returns today's date in 'YYYY-MM-DD' format.

    Returns:
        str: Today's date in 'YYYY-MM-DD' format.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# COMMAND ----------

# DBTITLE 1,Test python function
today = get_todays_date()
today

# COMMAND ----------

# DBTITLE 1,Register python function to Unity Catalog
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# this will deploy the tool to UC, automatically setting the metadata in UC based on the tool's docstring & typing hints
python_tool_uc_info = client.create_python_function(func=get_todays_date, catalog=catalog_name, schema=schema_name, replace=True)

# the tool will deploy to a function in UC called `{catalog}.{schema}.{func}` where {func} is the name of the function
# Print the deployed Unity Catalog function name
#print(f"Deployed Unity Catalog function name: {python_tool_uc_info.full_name}")

# COMMAND ----------

# DBTITLE 1,Let's take a look at our created functions
from IPython.display import display, HTML

# Retrieve the Databricks host URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Create HTML link to created functions
html_link = f'<a href="https://{workspace_url}/explore/data/functions/{catalog_name}/{schema_name}/get_order_history" target="_blank">Go to Unity Catalog to see Registered Functions</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now lets go over to the AI Playground to see how we can use these functions and assemble our first Agent!
# MAGIC
# MAGIC ### The AI Playground can be found on the left navigation bar under 'AI/ML' or you can use the link created below

# COMMAND ----------

# DBTITLE 1,Create link to AI Playground
# Create HTML link to AI Playground
html_link = f'<a href="https://{workspace_url}/ml/playground" target="_blank">Go to AI Playground</a>'
display(HTML(html_link))
