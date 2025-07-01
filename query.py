import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# These should match the settings used in bot.py
index_name = "langchainvector"
embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

# Reconnect to the existing Pinecone vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

def query_pdf(user_query, k=3):
    """
    Query the Pinecone vector store for the most relevant PDF chunks to the user_query.
    Returns a list of (page_content, metadata) tuples.
    """
    results = vectorstore.similarity_search(user_query, k=k)
    return [(doc.page_content, doc.metadata) for doc in results]

if __name__ == "__main__":
    print("Ask a question about the PDF:")
    user_query = input().strip()
    results = query_pdf(user_query)
    print("\nTop results:")
    for i, (content, meta) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(content)
