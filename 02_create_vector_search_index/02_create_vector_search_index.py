# Databricks notebook source
# MAGIC %md
# MAGIC # Building a knowledge base using Mosaic AI Vector search 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq langchain langgraph databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Check the library versions. Modify to lock-in with the project version
import pkg_resources

# List of packages to check
packages_to_check = [
    "langchain", 
    "langgraph", 
    "databricks-langchain", 
    "pydantic", 
    "databricks-agents", 
    "unitycatalog-langchain", 
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

spark.sql(f"USE CATALOG `{catalog_name}`")
spark.sql(f"USE SCHEMA  `{schema_name}`")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT current_catalog(), current_schema()

# COMMAND ----------

spark.sql(f"""
          SELECT * FROM {catalog_name}.{schema_name}.product_docs LIMIT 5
          """).display()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 2/ Create our vector search table
# MAGIC
# MAGIC ### 2.1/ Vector search Endpoints
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-2.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC Vector search endpoints are entities where your indexes will live. Think about them as entry point to handle your search request. 
# MAGIC
# MAGIC Let's start by creating our first Vector Search endpoint. Once created, you can view it in the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

import time

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

print(VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-3.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC
# MAGIC ### 2.2/ Creating the Vector Search Index
# MAGIC
# MAGIC Once the endpoint is created, all we now have to do is to as Databricks to create the index on top of the existing table. 
# MAGIC
# MAGIC You just need to specify the text column and our embedding foundation model (`GTE`).  Databricks will build and synchronize the index automatically for us.
# MAGIC
# MAGIC Note that Databricks provides 3 type of vector search:
# MAGIC
# MAGIC * **Managed embeddings**: Databricks creates the embeddings for you from a text field and Databricks synchronize the Delta table to your index (what we'll use)
# MAGIC * **Self managed embeddings**: You compute the embeddings yourself and save them to your Delta table  and Databricks synchronize the Delta table to your index
# MAGIC * **Direct access**: you manage the VS indexation yourself (no Delta table)
# MAGIC
# MAGIC This can be done using the API, or in a few clicks within the Unity Catalog Explorer menu:
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/index_creation.gif?raw=true" width="600px">
# MAGIC

# COMMAND ----------

# Before inserting indexes, we need to enable CDC. 
spark.sql(f"""
          ALTER TABLE {catalog_name}.{schema_name}.product_docs SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
          """)

# COMMAND ----------

# Uncomment this if you are starting a new table
# %sql
# CREATE TABLE IF NOT EXISTS product_docs (
#   id BIGINT GENERATED ALWAYS AS IDENTITY,
#   product_name STRING,
#   title STRING,
#   content STRING,
#   doc_uri STRING)
#   TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

def wait_for_model_serving_endpoint_to_be_ready(ep_name):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # TODO make the endpoint name as a param
    # Wait for it to be ready
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(ep_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

# COMMAND ----------

# DBTITLE 1,Get the Latest Return in the Processing Queue
from databricks.sdk import WorkspaceClient

#The table we'd like to index
source_table_fullname = f"{catalog_name}.{schema_name}.product_docs"
# Where we want to store our index
vs_index_fullname = f"{catalog_name}.{schema_name}.product_docs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="product_id",
    embedding_source_column='indexed_doc', #The column containing our text
    embedding_model_endpoint_name='databricks-gte-large-en' #The embedding endpoint used to create the embeddings
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2.3/ Try our VS index: searching for relevant content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your table with the index.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also support a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

# DBTITLE 1,Create a function registered to Unity Catalog
question = "What is the warrenty for SoundWave Pro X5?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["product_id", "indexed_doc"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs
