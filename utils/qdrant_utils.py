# utils/qdrant_utils.py
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

def collection_exists(collection_name):
    """Check if a collection exists"""
    url = f"{QDRANT_URL}/collections/{collection_name}"
    try:
        response = requests.get(url, headers=HEADERS)
        return response.status_code == 200
    except:
        return False

def create_collection(collection_name, vector_size):
    """Create a new collection"""
    url = f"{QDRANT_URL}/collections/{collection_name}"
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }
    response = requests.put(url, headers=HEADERS, json=payload)
    if response.status_code not in [200, 409]:  # 409 means collection already exists
        response.raise_for_status()
    return response.json()

def create_source_index(collection_name):
    """Create an index for the 'source' field to enable filtering"""
    url = f"{QDRANT_URL}/collections/{collection_name}/index"
    payload = {
        "field_name": "source",
        "field_schema": "keyword"
    }
    response = requests.put(url, headers=HEADERS, json=payload)
    if response.status_code not in [200, 409]:  # 409 means index already exists
        response.raise_for_status()
    return response.json()

def upsert_vectors(collection_name, points):
    """Insert or update vectors in the collection"""
    url = f"{QDRANT_URL}/collections/{collection_name}/points"
    payload = {"points": points}
    response = requests.put(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()