# Databricks notebook source
# MAGIC %md
# MAGIC ## Load configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo in a different schema.

# COMMAND ----------

catalog_name = "databricks_workshop"
schema_name = "jywu"
VECTOR_SEARCH_ENDPOINT_NAME="databricks_workshop_vector_search"

# COMMAND ----------

print(f"import catalog_name as {catalog_name}")
print(f"import schema_name as {schema_name}")
print(f"import VECTOR_SEARCH_ENDPOINT_NAME as {VECTOR_SEARCH_ENDPOINT_NAME}")