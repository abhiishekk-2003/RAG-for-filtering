# Qdrant_utils.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": QDRANT_API_KEY
}

def create_collection(collection_name, vector_size):
    url = f"{QDRANT_URL}/collections/{collection_name}"
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }
    response = requests.put(url, headers=HEADERS, json=payload)
    return response.json()

def upsert_vectors(collection_name, points):
    url = f"{QDRANT_URL}/collections/{collection_name}/points"
    payload = {"points": points}
    response = requests.put(url, headers=HEADERS, json=payload)
    return response.json()
