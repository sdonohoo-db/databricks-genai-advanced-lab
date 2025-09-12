# Databricks notebook source
# MAGIC %md
# MAGIC # Create an unstructured data pipeline for GenAI retrievers
# MAGIC
# MAGIC Before building vector search indexes, it's essential to first prepare your unstructured data through a dedicated data engineering step. This involves ingesting, cleaning, and transforming raw documents into a structured format—typically chunked, metadata-enriched, and stored in Delta tables. This foundation ensures your GenAI retrievers operate on high-quality, queryable content.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

query = f"""
SELECT path
FROM READ_FILES('/Volumes/{catalog_name}/{schema_name}/pdfs/', format => 'binaryFile')
limit 5
"""

spark.sql(query).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extracting the information using `ai_parse_document`
# MAGIC Databricks provides a builtin `ai_parse_document` function, leveraging AI to analyze and extract PDF information as text. This makes it super easy to ingest unstructured information!
# MAGIC
# MAGIC - Easy to adopt. Recommended for the POC and long-term. 
# MAGIC - This AI function can work with PDF, JPG, and PNG. The product team recently added PPTX as well. 
# MAGIC
# MAGIC https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document?language=SQL

# COMMAND ----------

spark.sql(f"""
          SELECT ai_parse_document(content) AS parsed_document
          FROM READ_FILES('/Volumes/{catalog_name}/{schema_name}/pdfs/', format => 'binaryFile') LIMIT 2
          """).display()

# COMMAND ----------

spark.sql(f"""
          WITH corpus AS (
            SELECT
                path,
                ai_parse_document(content) AS parsed
            FROM
                READ_FILES('/Volumes/{catalog_name}/{schema_name}/pdfs/', format => 'binaryFile')
            )
            SELECT
            path,
            parsed:document:pages,
            parsed:document:elements,
            parsed:corrupted_data,
            parsed:error_status,
            parsed:metadata
            FROM corpus
          """).display()

# COMMAND ----------

spark.sql(f"""
          WITH corpus AS (
            SELECT
                path,
                ai_parse_document(content) AS parsed
            FROM
                READ_FILES('/Volumes/{catalog_name}/{schema_name}/pdfs/', format => 'binaryFile')
            )
            SELECT 
            path as doc_uri,
            array_join(transform(parsed:document.pages::ARRAY<STRUCT<content:STRING>>, x -> x.content), '\n') AS content
            FROM corpus
          """).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Python UDF
# MAGIC Databricks' notebook [template](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/unstructured-data-pipeline.html) to parse, chunk and create index. PDF, DOCX, HTML, TXT, MD, JSON are included in the template and the logic can be extended.
# MAGIC - Recommended for quick trial and error as it requires additional code.
# MAGIC - This is a more hard-coded way but more robust as well. 
# MAGIC - Search for `file_parser` function to take a closer look.
