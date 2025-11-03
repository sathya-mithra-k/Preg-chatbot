from os import getenv

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

# Get your index
index = pc.Index("langchainvector")

stats = index.describe_index_stats()
print(stats)
