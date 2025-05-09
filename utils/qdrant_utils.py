import requests

QDRANT_URL = "https://ee883337-0943-4db9-99d5-9bd1365c2541.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.atlxdgC9NOgHHQKh4tiLlTSTcJIVNYzaNczuANYaPf8"

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
