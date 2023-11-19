import os
import openai

openai.api_version = '2023-05-15'
openai.api_base = "https://trefoil.openai.azure.com/"
openai.api_type = 'azure'
openai.api_key = "deec54ba420e44bbac7fbc640b724fd4"


deployment_id = "text-embedding-ada-002"
embeddings = openai.Embedding.create(deployment_id=deployment_id,
                                     input="The food was delicious and the waiter...")
                                
print(embeddings)