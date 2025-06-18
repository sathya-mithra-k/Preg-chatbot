from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key= os.getenv("GEMINI_API_KEY"))

def read(direc):
    loader = PyPDFLoader(direc)
    document = loader.load()
    return document

doc = read(r"C:\Users\sathy\Preg\Preg-chatbot\Journey_of_The_First_1000_Days.pdf")

def chunking(docs,chunk_size=256,chunk_overlap = 50):
    split = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    doc = split.split_documents(docs)
    return docs

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/sathy/Downloads/preg-bot-463313-44ca92552e99.json"
embedding = GoogleGenerativeAIEmbeddings(model= "models/gemini-embedding-exp-03-07")

#vector = embedding.embed_query("How are you")
#print(len(vector))

api_key = "pcsk_3rvtmR_3uGwGmzHeMK4YWt3mQMouSQjf8MhogGxbTVyuEJN7h8FLwq4ft37UCjjNDdRN1f"
index_name = "langchainvector"
dimension = 3072

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',  
        spec=ServerlessSpec(
            cloud='aws',  
            region='us-east-1' 
        )
    )

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(index, embedding=your_embedding_function)