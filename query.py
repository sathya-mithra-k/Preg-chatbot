import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import openai
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# These should match the settings used in bot.py
index_name = "langchainvector"
embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

# Reconnect to the existing Pinecone vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

def query_pdf(user_query, k=2):
    results = vectorstore.similarity_search(user_query, k=k)
    return [(doc.page_content, doc.metadata) for doc in results]

def generate_answer(context, question, model="gpt-3.5-turbo"):
    prompt = f"""
You are a helpful assistant. Use the following context from a PDF to answer the user's question. Be concise and only use information from the context. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("Ask a question about the PDF:")
    user_query = input().strip()
    results = query_pdf(user_query, k=3)
    context = "\n---\n".join([content for content, _ in results])
    #print("\nTop retrieved context:\n", context)
    answer = generate_answer(context, user_query)
    print("\nAnswer:\n", answer)
