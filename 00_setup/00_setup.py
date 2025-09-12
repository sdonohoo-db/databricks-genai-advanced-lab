# Databricks notebook source
# MAGIC %md
# MAGIC # Initialize Catalog, Schema, Data, Code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Catalog, Schema

# COMMAND ----------

# Create a widget to input the catalog name
dbutils.widgets.text("catalog_name", "databricks_workshop", "Catalog Name")

# Retrieve the catalog name from the widget
catalog_name = dbutils.widgets.get("catalog_name")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
#import yaml
import os

# Use the workspace client to retrieve information about the current user
w = WorkspaceClient()
emails = w.current_user.me().emails
user_email = next(e.value for e in emails if e.primary)
username = user_email.split("@")[0].replace(".", "_") # only letters and underscores
print(f"Proceed to the next cell and set the schema as {username}")

# COMMAND ----------

# Catalog and schema have been automatically created thanks to lab environment
schema_name = f"{username}"

# Create the schema if it does not already exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- verify the catalog and schema
# MAGIC SELECT current_catalog(), current_schema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download data from Github to Volume

# COMMAND ----------

# download from Github
import requests
import pandas as pd
import io
import time

# config file paths
base_url = "https://raw.githubusercontent.com/jiayi-wu-3150/databricks-genai-advanced-lab/main/data/"
csv_files = {
    "product_docs": f"{base_url}/csvs/product_docs.csv",
    "customer_services": f"{base_url}/csvs/customer_services.csv",
    "policies": f"{base_url}/csvs/policies.csv", 
    "inventories": f"{base_url}/csvs/inventories.csv",
} 
pdf_files = {
    "BrownBox_SwiftWatch_X500_Manual.pdf": f"{base_url}/pdfs/BrownBox_SwiftWatch_X500_Manual.pdf",
    "SoundWave_X5_Pro_Headphones_Manual.pdf": f"{base_url}/pdfs/SoundWave_X5_Pro_Headphones_Manual.pdf",
}

spark.sql(f"CREATE VOLUME  IF NOT EXISTS `{catalog_name}`.`{schema_name}`.`raw_data`")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS `{catalog_name}`.`{schema_name}`.`pdfs`")

csv_dir = f"/Volumes/{catalog_name}/{schema_name}/raw_data"
pdf_dir = f"/Volumes/{catalog_name}/{schema_name}/pdfs"

# Download and load each CSV file
for table_name, url in csv_files.items():
    # Download CSV data
    response = requests.get(url)
    response.raise_for_status()
    
    # Save original CSV bytes into the raw_file volume
    csv_path = os.path.join(csv_dir, f"{table_name}.csv")
    with open(csv_path, "wb") as f:
        f.write(response.content)

    # Save CSV into pandas DataFrame, Convert to Spark DataFrame and write to table
    df = pd.read_csv(io.StringIO(response.text))
    spark_df = spark.createDataFrame(df)
    spark_df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")
    print(f"Table {table_name} created successfully; CSV saved to {csv_path}")

for file_name, url in pdf_files.items():
    # Download PDF data
    response = requests.get(url)
    response.raise_for_status()
    
    # Save PDF to the specified volume
    pdf_path = os.path.join(pdf_dir, file_name)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    print(f"PDF saved successfully: {pdf_path}")

# Quick check 
display(dbutils.fs.ls(f"/Volumes/{catalog_name}/{schema_name}/raw_data"))
display(dbutils.fs.ls(f"/Volumes/{catalog_name}/{schema_name}/pdfs"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the template code

# COMMAND ----------

# update code from jywu to specified schema_name
import os
root = os.getcwd().rsplit("/", 1)[0] + '/'
current_file = os.getcwd().rsplit("/", 1)[-1]

for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith((".py", ".ipynb")) and not filename.startswith(current_file):
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if "jywu" in content:
                new_content = content.replace("jywu", f"{schema_name}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated: {filepath}")