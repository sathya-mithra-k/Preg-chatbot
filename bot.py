import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

test_vector = embedding.embed_query("Hello world")
print("Embedding dimension:", len(test_vector))

pdf_path = r"C:\Users\sathy\Preg\Preg-chatbot\JOURNEY OF THE FIRST 100 DAYS.pdf"
documents = load_and_split(pdf_path)

pc = Pinecone(api_key = pinecone_api_key)

index_name = "langchainvector"
dimension = 3072 

spec = ServerlessSpec(
    cloud = 'aws', region = 'us-east-1'
)

vectorstore = PineconeVectorStore(index_name = index_name, embedding = embedding)

vectorstore.add_documents(documents)


