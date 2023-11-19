

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


## Setup Azure OpenAI
We'll start as usual by defining our Azure OpenAI service API key and endpoint details, specifying the model deployment we want to use and then we'll initiate a connection to the Azure OpenAI service.

