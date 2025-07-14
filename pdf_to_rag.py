#%% Librerías
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticsearchStore

#%% Variables de entorno
OPENAI_API_KEY = "sk-proj-"

#%% Splitear el documento PDF

path_pdf = fr''
loader = PyPDFLoader(path_pdf)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )
    
docs = text_splitter.split_documents(data)

#%% Realizar el cargue de los datos en Elasticsearch

embeddings = OpenAIEmbeddings()

db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://0.0.0.0:9200",
    es_user="elastic",
    es_password="WNzA1O*Daecj59qNcJ4V",
    index_name="consejos_de_ahorro", #Nombre de la colección específica
)

db.client.indices.refresh(index="consejos_de_ahorro")

print(f"PDF procesado en {len(docs)} chunks usando RecursiveCharacterTextSplitter.")
