
# AI Orchestration with Azure Cognitive Search
In this lab, we will do a deeper dive around the Azure Cognitive Search vector store and different ways to interact with it.

Create Azure Cognitive Search Vector Store in Azure
First, we need to create an Azure Cognitive Search service in Azure, which will act as a vector store. We'll use the Azure CLI to do this.

<details>
  <summary>:white_check_mark: See solution!</summary>

```
 RESOURCE_GROUP="azure-cognitive-search-rg"
 LOCATION="westeurope"
 NAME="acs-vectorstore-<INITIALS>"
 !az group create --name $RESOURCE_GROUP --location $LOCATION
 !az search service create -g $RESOURCE_GROUP -n $NAME -l $LOCATION --sku Basic --partition-count 1 --replica-count 1
```
</details>

___

## Setup Azure OpenAI
We'll start as usual by defining our Azure OpenAI service API key and endpoint details, specifying the model deployment we want to use and then we'll initiate a connection to the Azure OpenAI service.

<details>
  <summary>:white_check_mark: See solution!</summary>

```
 import os
 from dotenv import load_dotenv

 # Load environment variables
 if load_dotenv():
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
 else: 
    print("No file .env found")

 openai_api_type = os.getenv("OPENAI_API_TYPE")
 openai_api_key = os.getenv("OPENAI_API_KEY")
 openai_api_base = os.getenv("OPENAI_API_BASE")
 openai_api_version = os.getenv("OPENAI_API_VERSION")
 deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
 embedding_name = os.getenv("OPENAI_EMBEDDING_DEPLOYMENTE")
 acs_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
 acs_endpoint_name = os.getenv("AAZURE_SEARCH_ENDPOINT")
 acs_index_name = "give-an index name"
 acs_api_key = os.getenv("AAZURE_SEARCH_KEY")
```
</details>

## Load the movies data 
Load the data from the movies.csv file using the Langchain CSV document loader.
<details>
  <summary>:white_check_mark: See solution!</summary>

```
 from langchain.document_loaders.csv_loader import CSVLoader

 # Movie Fields in CSV
 # id,original_language,original_title,popularity,release_date,vote_average,vote_count,genre,overview,revenue,runtime,tagline
 loader = CSVLoader(file_path='./movies.csv', source_column='original_title', encoding='utf-8', csv_args={'delimiter':',', 'fieldnames': ['id', 'original_language', 'original_title', 'popularity', 'release_date', 'vote_average', 'vote_count', 'genre', 'overview', 'revenue', 'runtime', 'tagline']})
 data = loader.load()
 data = data[1:51] # reduce dataset if you want
 print('Loaded %s movies' % len(data))
```
</details>

## Create an embedding 
Next, we will create an Azure OpenAI embedding and completion deployments in order to create the vector representation of the movies so we can start asking our questions.
<details>
  <summary>:white_check_mark: See solution!</summary>

```
 from langchain.embeddings.openai import OpenAIEmbeddings
 from langchain.chat_models import AzureChatOpenAI

 # Create an Embeddings Instance of Azure OpenAI
 embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=embedding_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    embedding_ctx_length=8191,
    chunk_size=1000,
    max_retries=6
 )

 # Create a Completion Instance of Azure OpenAI
 llm = AzureChatOpenAI(
    model="gpt-3.5-turbo",
    deployment_name = deployment_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    temperature=0.7,
    max_retries=6,
    max_tokens=4000
 )
 print('Completed creation of embedding and completion instances.')
 ```
</details>

## Load Movies into Azure Cognitive Search
Next, we'll create the Azure Cognitive Search index, embed the loaded movies from the CSV file, and upload the data into the newly created index. Depending on the number of movies loaded and rate limiting, this might take a while to do the embeddings so be patient.

<details>
  <summary>:white_check_mark: See solution!</summary>

```
 from azure.core.credentials import AzureKeyCredential
 from azure.search.documents.indexes import SearchIndexClient
 from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchIndex,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField,
    SearchField,
    SemanticSettings,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
)

 # Let's Create the Azure Cognitive Search Index
 index_client = SearchIndexClient(
    acs_endpoint_name,
    AzureKeyCredential(acs_api_key)
)
 # Movie Fields in CSV
 # id,original_language,original_title,popularity,release_date,vote_average,vote_count,genre,overview,revenue,runtime,tagline
 fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="tagline", type=SearchFieldDataType.String),
    SearchableField(name="popularity", type=SearchFieldDataType.Double, sortable=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config"),
 ]

 # Configure Vector Search Configuration
 vector_search = VectorSearch(
    algorithm_configurations=[
        HnswVectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw",
            parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"
            }
        )
    ]
)

 # Configure Semantic Configuration
 semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="title"),
        prioritized_keywords_fields=[SemanticField(field_name="title"), SemanticField(field_name="tagline")],
        prioritized_content_fields=[SemanticField(field_name="content")]
    )
)

 # Create the semantic settings with the configuration
 semantic_settings = SemanticSettings(configurations=[semantic_config])

 # Create the search index with the desired vector search and semantic configurations
 index = SearchIndex(
    name=acs_index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_settings=semantic_settings
)
 result = index_client.create_or_update_index(index)
 print(f'The {result.name} index was created.')
 ```
</details>

##  Create the document structure

Next, create the document structure needed to upload the data into the Azure Cognitive Search index.

<details>
  <summary>:white_check_mark: See solution!</summary>

```
 # Now that the index is created, let's load the documents into it.

 import uuid

 # Let's take a quick look at the data structure of the CSVLoader
 print(data[0])
 print(data[0].metadata['source'])
 print("----------")

 # Generate Document Embeddings for page_content field in the movies CSVLoader dataset using Azure OpenAI
 items = []
 for movie in data:
    content = movie.page_content
    items.append(dict([("id", str(uuid.uuid4())), ("title", movie.metadata['source']), ("content", content), ("content_vector", embeddings.embed_query(content))]))

 # Print out a sample item to validate the updated data structure.
 # It should have the id, content, and content_vector values.
 print(items[0])
 print(f"Movie Count: {len(items)}")

```
</details>

##  Create the document structure

Next, upload the movie documents in the newly created structure to the Azure Cognitive Search index.
<details>
  <summary>:white_check_mark: See solution!</summary>

```
 # Upload movies to Azure Cognitive Search index. 
 from azure.search.documents.models import Vector
 from azure.search.documents import SearchClient

 # Insert Text and Embeddings into the Azure Cognitive Search index created.
 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
)
 result = search_client.upload_documents(items)
 print("Successfully added documents to Azure Cognitive Search index.")
 print(f"Uploaded {len(data)} documents")
```
</details>

## Vector Store Searching using Azure Cognitive Search

Now that we have the movies loaded into Azure Cognitive Search, apply some different types of searches using the Azure Cognitive Search SDK.
* First, a plain vanilla text search, no vectors or embeddings.
* Then a vector search that uses the embeddings we created and inserted into `content_vector` field in the index.

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # First, let's do a plain vanilla text search, no vectors or embeddings.
 query = "What are the best 80s movies I should look at?"

 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )

 # Execute the search
 results = list(search_client.search(
    search_text=query,
    include_total_count=True,
    top=5
))

 # Print count of total results.
 print(f"Returned {len(results)} results using only text-based search.")
 print("----------")
 # Iterate over Results
 # Index Fields - id, content, content_vector
 for result in results:
    print("Movie: {}".format(result["content"]))
    print("----------")
 ```
</details>


* Then a vector search that uses the embeddings we created and inserted into `content_vector` field in the index.

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # Now let's do a vector search that uses the embeddings we created and inserted into content_vector field in the index.
 query = "What are the best 80s movies I should look at?"
 
 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )
 
# You can see here that we are getting the embedding representation of the query.
 vector = Vector(
    value=embeddings.embed_query(query),
    k=5,
    fields="content_vector"
)

 # Execute the search
 results = list(search_client.search(
    search_text="",
    include_total_count=True,
    vectors=[vector],
    select=["id", "content", "title"],
 ))

 # Print count of total results.
 print(f"Returned {len(results)} results using only vector-based search.")
 print("----------")
 # Iterate over results and print out the content.
 for result in results:
    print(result["title"])
    print("----------")
 ```
 </details>


### Search Score 1

Did that return what you expected? Probably not, let's dig deeper to see why.
Please reproduce the same search again, but this time let's return the Search Score so we can see the value returned by the cosine similarity vector store calculation.

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # Try again, but this time let's add the relevance score to maybe see why
 query = "What are the best 80s movies I should look at?"
 
 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )

 # You can see here that we are getting the embedding representation of the query.
 vector = Vector(
    value=embeddings.embed_query(query),
    k=5,
    fields="content_vector"
 )

 # Execute the search
 results = list(search_client.search(
    search_text="",
    include_total_count=True,
    vectors=[vector],
    select=["id", "content", "title"],
 ))

 # Print count of total results.
 print(f"Returned {len(results)} results using vector search.")
 print("----------")
 # Iterate over results and print out the id and search score.
 for result in results:  
    print(f"Id: {result['id']}")
    print(f"Id: {result['title']}")
    print(f"Score: {result['@search.score']}")
    print("----------")

 ```
 </details>

### Search Score 2 
 If you look at the Search Score you will see the relevant ranking of the closest vector match to the query inputted. The lower the score the farther apart the two vectors are. 
 Please change the search term and see if we can get a higher Search Score which means a higher match and closer vector proximity.
 **NOTE:** As you have seen from the results, different inputs can return different results, it all depends on what data is in the Vector Store. The higher the score the higher the likelihood of a match.

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # Try again, but this time let's add the relevance score to maybe see why
 query = "Who are the actors in the movie Hidden Figures?"

 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )

 # You can see here that we are getting the embedding representation of the query.
 vector = Vector(
    value=embeddings.embed_query(query),
    k=5,
    fields="content_vector"
 )

 # Execute the search
 results = list(search_client.search(
    search_text="",
    include_total_count=True,
    vectors=[vector],
    select=["id", "content", "title"],
 ))

 # Print count of total results.
 print(f"Returned {len(results)} results using vector search.")
 print("----------")
 # Iterate over results and print out the id and search score.
 for result in results:  
    print(f"Id: {result['id']}")
    print(f"Id: {result['title']}")
    print(f"Score: {result['@search.score']}")
    print("----------")
 ```
 </details>

## Hybrid Searching using Azure Cognitive Search

What is Hybrid Search? The search is implemented at the field level, which means you can build queries that include vector fields and searchable text fields. The queries execute in parallel and the results are merged into a single response. Optionally, add semantic search, currently in preview, for even more accuracy with L2 reranking using the same language models that power Bing.

**NOTE:** Hybrid Search is a key value proposition of Azure Cognitive Search in comparison to vector only data stores. Click Hybrid Search for more details.

### Hybrid Search 1

#### Try our original query again using Hybrid Search (ie. Combination of Text & Vector Search)

`query = "What are the best 80s movies I should look at?"`

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # Hybrid Search
 # Let's try our original query again using Hybrid Search (ie. Combination of Text & Vector Search)
 query = "What are the best 80s movies I should look at?"

 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )

 # You can see here that we are getting the embedding representation of the query.
 vector = Vector(
    value=embeddings.embed_query(query),
    k=5,
    fields="content_vector"
)

 # Notice we also fill in the search_text parameter with the query.
 results = list(search_client.search(
    search_text=query,
    include_total_count=True,
    top=10,
    vectors=[vector],
    select=["id", "content", "title"],
 ))

 # Print count of total results.
 print(f"Returned {len(results)} results using vector search.")
 print("----------")
 # Iterate over results and print out the id and search score.
 for result in results:  
    print(f"Id: {result['id']}")
    print(result['title'])
    print(f"Hybrid Search Score: {result['@search.score']}")
    print("----------")
 ```
 </details>

### Hybrid Search 2
#### Try our more specific query again to see the difference in the score returned.
`query = "Who are the actors in the movie Hidden Figures?"`


<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )

 # You can see here that we are getting the embedding representation of the query.
 vector = Vector(
    value=embeddings.embed_query(query),
    k=5,
    fields="content_vector"
 )

 # -----
 # Notice we also fill in the search_text parameter with the query along with the vector.
 # -----
 results = list(search_client.search(
    search_text=query,
    include_total_count=True,
    top=10,
    vectors=[vector],
    select=["id", "content", "title"],
 ))

 # Print count of total results.
 print(f"Returned {len(results)} results using hybrid search.")
 print("----------")
 # Iterate over results and print out the id and search score.
 for result in results:  
    print(f"Id: {result['id']}")
    print(f"Title: {result['title']}")
    print(f"Hybrid Search Score: {result['@search.score']}")
    print("----------")
 ```
 </details>


## Bringing it All Together with Retrieval Augmented Generation (RAG) + Langchain (LC)

Now that we have our Vector Store setup and data loaded, we are now ready to implement the RAG pattern using AI Orchestration. At a high-level, the following steps are required:

* Ask the question

* Create Prompt Template with inputs

* Get Embedding representation of inputted question

* Use embedded version of the question to search Azure Cognitive Search (ie. The Vector Store)

* Inject the results of the search into the Prompt Template & Execute the Prompt to get the completion

###  Setup Langchain and  Create an Embeddings Instance of Azure OpenAI
<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 # Implement RAG using Langchain (LC)

 from langchain.embeddings.openai import OpenAIEmbeddings
 from langchain.chat_models import AzureChatOpenAI
 from langchain.chains import LLMChain

 # Setup Langchain
 # Create an Embeddings Instance of Azure OpenAI
 embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment=embedding_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    embedding_ctx_length=8191,
    chunk_size=1000,
    max_retries=6
 )
 # Create a Completion Instance of Azure OpenAI
 llm = AzureChatOpenAI(
    model="gpt-3.5-turbo",
    deployment_name = deployment_name,
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    temperature=0.7,
    max_retries=6,
    max_tokens=4000
 )
 ```
 </details>

###  Create a prompt template to ask the question

<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 question = "List the movies about ships on the water."

 # Create a prompt template with variables, note the curly braces
 from langchain.prompts import PromptTemplate
 prompt = PromptTemplate(
    input_variables=["original_question","search_results"],
    template="""
    Question: {original_question}

    Do not use any other data.
    Only use the movie data below when responding.
    {search_results}
    """,
 )
 # Get Embedding for the original question
 question_embedded=embeddings.embed_query(question)
 ```
 </details>

## Search Vector Store
### Build the Prompt and Execute against the Azure OpenAI to get the completion
<details>
  <summary>:white_check_mark: See solution!</summary>

 ```
 search_client = SearchClient(
    acs_endpoint_name,
    acs_index_name,
    AzureKeyCredential(acs_api_key)
 )
 vector = Vector(
    value=question_embedded,
    k=5,
    fields="content_vector"
)
 results = list(search_client.search(
    search_text="",
    include_total_count=True,
    vectors=[vector],
    select=["title"], 
 ))

 # Build the Prompt and Execute against the Azure OpenAI to get the completion

 chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
 response = chain.run({"original_question": question, "search_results": results})
 print(response)
 ```
 </details>
