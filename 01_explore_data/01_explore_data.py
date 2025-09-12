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

# we will use this table to create vector search indexes
spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.product_docs limit 5").display()

# COMMAND ----------

# we will use this table create SQL functions
spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.customer_services limit 5").display()

# COMMAND ----------

# we will use this table create SQL functions
spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.policies limit 5").display()