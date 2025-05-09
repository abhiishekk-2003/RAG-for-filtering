# embed_query.py
from utils.embedder import get_embedding

def embed_query(text: str) -> list:
    embedding = get_embedding(f"query: {text}")
    print("Query embedding ready. Vector size:", len(embedding))
    return embedding
