# utils/retriever.py
import requests
import json

QDRANT_URL = "https://ee883337-0943-4db9-99d5-9bd1365c2541.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.atlxdgC9NOgHHQKh4tiLlTSTcJIVNYzaNczuANYaPf8"
COLLECTION_NAME = "Doctor's data"  

def search_qdrant(vector, top_k=5):
    if not isinstance(vector, list) or not all(isinstance(x, float) for x in vector):
        raise ValueError("Invalid vector! Must be a list of floats.")

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    payload = {
        "vector": vector,
        "top": top_k,
        "with_payload": True
    }


    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        headers=headers,
        json=payload,
    )

    response.raise_for_status()
    return response.json()["result"]
