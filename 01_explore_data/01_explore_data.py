# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Exploration
# MAGIC
# MAGIC Let's explore the data before jumping into transformation. Understanding the structure and content will help guide the preprocessing steps.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

spark.sql(f"USE CATALOG `{catalog_name}`")
spark.sql(f"USE SCHEMA  `{schema_name}`")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT current_catalog(), current_schema()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- we will use this table to create vector search indexes
# MAGIC SELECT * FROM databricks_workshop.jywu.product_docs
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- we will use this table create SQL functions
# MAGIC SELECT * FROM databricks_workshop.jywu.customer_services
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- we will use this table create SQL functions
# MAGIC SELECT * FROM databricks_workshop.jywu.policies
# MAGIC LIMIT 5