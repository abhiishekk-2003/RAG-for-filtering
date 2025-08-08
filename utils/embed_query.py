# utils/embed_query.py
from utils.embedder import get_embedding

def embed_query(text: str) -> list:
    """Embed a query text with the 'query:' prefix for better retrieval"""
    embedding = get_embedding(f"query: {text}")
    print("Query embedding ready. Vector size:", len(embedding))
    return embedding