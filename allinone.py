# Combined RAG Pipeline with Doctor Info Extraction

import os
import json
import uuid
import requests
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Load environment variables
load_dotenv()

# Constants and config
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_SIZE = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_here")

# Headers
HF_HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}
QDRANT_HEADERS = {
    "Content-Type": "application/json",
    "api-key": QDRANT_API_KEY
}

# Embedding Function
def get_embedding(text):
    payload = {"inputs": [text]}
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    if response.status_code != 200:
        print("Embedding Error:", response.text)
        response.raise_for_status()
    result = response.json()
    return result[0] if isinstance(result, list) else result

# Embed Query
def embed_query(text):
    embedding = get_embedding(f"query: {text}")
    print("Query embedding ready. Vector size:", len(embedding))
    return embedding

# LLM Query Function
def ask_llama3(context, question):
    prompt = f"""
You are a helpful assistant specialized in answering based only on the provided context.
If the answer is not found in the context, respond strictly with: \"I cannot answer such questions.\"

Think step-by-step:
1. Understand the question.
2. Locate the relevant details in the context.
3. Generate a concise and factual response.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
"""
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }
    response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()
    fallback_message = "I cannot answer such questions."
    return fallback_message if not answer or fallback_message.lower() in answer.lower() else answer

# Load Text from File
def load_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    contents = []

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "webSearchResults" in data:
                for entry in data["webSearchResults"]:
                    if isinstance(entry, list):
                        for item in entry:
                            if isinstance(item, dict) and "content" in item:
                                cleaned = item["content"].replace("\n", " ").strip()
                                if cleaned:
                                    contents.append(cleaned)
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            contents.append(" ".join([page.extract_text() or "" for page in pdf.pages]))
    elif ext == ".docx":
        doc = Document(file_path)
        contents.append("\n".join([para.text for para in doc.paragraphs]))
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                contents.append(text)
    return contents

# Chunking
def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# Embed and Prepare Chunks
def embed_chunks(contents, filename):
    points = []
    absolute_chunk_index = 0
    for content_idx, content in enumerate(contents):
        for relative_idx, chunk in enumerate(chunk_text(content)):
            embedding = get_embedding(chunk)
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            if len(embedding) != VECTOR_SIZE:
                raise ValueError("Invalid embedding length")
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "source": filename,
                    "content_id": f"{filename}_content_{content_idx+1}",
                    "chunk_index": relative_idx,
                    "absolute_index": absolute_chunk_index
                }
            })
            absolute_chunk_index += 1
    return points

# Format Context
def format_context(results):
    return "\n---\n".join([hit["payload"]["text"] for hit in results])

# Qdrant Search
def search_qdrant(vector, top_k=5):
    payload = {
        "vector": vector,
        "top": top_k,
        "with_payload": True
    }
    response = requests.post(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search", headers=QDRANT_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["result"]

# Upsert to Qdrant
def upsert_vectors(collection_name, points):
    url = f"{QDRANT_URL}/collections/{collection_name}/points"
    payload = {"points": points}
    response = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    return response.json()

# Create Qdrant Collection with Index
def create_collection(collection_name, vector_size):
    url = f"{QDRANT_URL}/collections/{collection_name}"
    payload = {
        "vectors": {
            "size": vector_size, 
            "distance": "Cosine"
        }
    }
    response = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    return response.json()

# Create Index for Source Field
def create_source_index(collection_name):
    """Create an index for the 'source' field to enable filtering"""
    url = f"{QDRANT_URL}/collections/{collection_name}/index"
    payload = {
        "field_name": "source",
        "field_schema": "keyword"
    }
    response = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    return response.json()

# Check if collection exists
def collection_exists(collection_name):
    """Check if a collection exists"""
    url = f"{QDRANT_URL}/collections/{collection_name}"
    try:
        response = requests.get(url, headers=QDRANT_HEADERS)
        return response.status_code == 200
    except:
        return False

# Alternative method to check for existing files without filtering
def get_all_sources_in_collection(client, collection_name):
    """Get all unique sources in the collection by scrolling through all points"""
    existing_sources = set()
    offset = None
    
    while True:
        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True
            )
            
            points, next_offset = scroll_result
            
            for point in points:
                if "source" in point.payload:
                    existing_sources.add(point.payload["source"])
            
            if next_offset is None:
                break
            offset = next_offset
            
        except Exception as e:
            print(f"Error scrolling collection: {e}")
            break
    
    return existing_sources

# Ingestion Pipeline
def run_ingestion_pipeline():
    # Check if collection exists, if not create it
    if not collection_exists(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        create_collection(COLLECTION_NAME, VECTOR_SIZE)
    
    # Create index for source field (this is idempotent - won't fail if index already exists)
    print(f"Creating index for 'source' field...")
    try:
        create_source_index(COLLECTION_NAME)
        print("Index created successfully")
    except Exception as e:
        print(f"Index creation info: {e}")  # This might fail if index already exists, which is okay
    
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Folder '{UPLOAD_FOLDER}' not found.")
        return

    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Get existing sources using the alternative method
    print("Checking for existing files...")
    existing_sources = get_all_sources_in_collection(client, COLLECTION_NAME)
    print(f"Found {len(existing_sources)} existing files in collection")

    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith((".pdf", ".docx", ".txt", ".json")):
            continue

        if filename in existing_sources:
            print(f"Skipping '{filename}' — already uploaded.\n")
            continue

        # Process and upload the file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing: {filename}")
        contents = load_text_from_file(file_path)
        vectors = embed_chunks(contents, filename)
        response = upsert_vectors(COLLECTION_NAME, vectors)
        print("Uploaded", len(vectors), "vectors for", filename)

# RAG Pipeline
def run_rag_pipeline(user_question):
    query_vector = embed_query(user_question)
    results = search_qdrant(query_vector)
    context = format_context(results)
    answer = ask_llama3(context, user_question)
    print("\n[DEBUG] Retrieved Context:\n", context)
    return answer

# Main CLI
if __name__ == "__main__":
    mode = input("Choose mode: [1] Ingest Files  [2] Ask a Question: ").strip()
    if mode == "1":
        run_ingestion_pipeline()
    elif mode == "2":
        question = input("Ask a question: ")
        answer = run_rag_pipeline(question)
        print("\nAnswer:\n", answer)
    else:
        print("Invalid option selected.")