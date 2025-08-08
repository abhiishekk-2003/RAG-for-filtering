# utils/retriever.py
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  

def search_qdrant(vector, top_k=5):
    """Search for similar vectors in Qdrant collection"""
    if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
        raise ValueError("Invalid vector! Must be a list of numbers.")

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    payload = {
        "vector": vector,
        "top": top_k,
        "with_payload": True
    }

    try:
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            headers=headers,
            json=payload,
        )

        response.raise_for_status()
        return response.json()["result"]
        
    except Exception as e:
        print(f"Error searching Qdrant: {str(e)}")
        return []