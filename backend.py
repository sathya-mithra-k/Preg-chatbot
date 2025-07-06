import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from typing import List, Optional

load_dotenv()

app = FastAPI(title="First 1000 Days Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("Missing required API keys. Please check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# These should match the settings used in bot.py
index_name = "langchainvector"
embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

# Reconnect to the existing Pinecone vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    include_context: bool = False

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    success: bool = True

class HealthResponse(BaseModel):
    status: str
    message: str

def query_pdf(user_query: str, k: int = 3):
    try:
        results = vectorstore.similarity_search(user_query, k=k)
        return [(doc.page_content, doc.metadata) for doc in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying PDF: {str(e)}")

def generate_answer(context: str, question: str, model: str = "gpt-3.5-turbo"):
    try:
        prompt = f"""
You are a helpful assistant who assists women from their 0th day of pregnancy until the child becomes 2 years old. Use the following context from a PDF to answer the user's question. Be detailed and only use information from the context. If the answer is not in the context, say you don't know.

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="First 1000 Days Chatbot API is running"
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer using RAG"""
    try:
        # Query the PDF for relevant context
        results = query_pdf(request.question, k=3)
        context = "\n---\n".join([content for content, _ in results])
        
        # Generate answer using the context
        answer = generate_answer(context, request.question)
        
        return QuestionResponse(
            answer=answer,
            context=context if request.include_context else None,
            success=True
        )
    except Exception as e:
        return QuestionResponse(
            answer=f"Sorry, I encountered an error: {str(e)}",
            success=False
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running and connected to Pinecone"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 