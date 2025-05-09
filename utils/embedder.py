# embedder.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

def get_embedding(text: str):
    payload = {"inputs": [text]}  # Text must be in a list

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("Hugging Face error:", response.text)
        response.raise_for_status()

    result = response.json()

    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]

    raise ValueError("Unexpected response format from Hugging Face API")
