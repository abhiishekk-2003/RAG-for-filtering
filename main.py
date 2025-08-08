# main.py
# For extracting text from uploaded files
import os
import pdfplumber
from utils.embedder import get_embedding
from utils.qdrant_utils import upsert_vectors, create_collection, create_source_index, collection_exists
import uuid
from docx import Document
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# For retrieving and asking a question
from utils.embed_query import embed_query
from utils.retriever import search_qdrant
from utils.formatter import format_context
from utils.groq_llm import ask_llama3

load_dotenv()

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_here")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_SIZE = 384  # GTE-small outputs 384-d vectors

def load_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    contents = []
    
    if ext == ".json":
        try:
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
            return contents

        except Exception as e:
            print(f"Failed to read JSON file: {file_path}\nError: {str(e)}")
            return []
        
    elif ext == ".pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    contents.append(text)
        except Exception as e:
            print(f"Failed to read PDF file: {file_path}\nError: {str(e)}")
            return []

    elif ext == ".docx":
        try:
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            if text.strip():
                contents.append(text)
        except Exception as e:
            print(f"Failed to read DOCX file: {file_path}\nError: {str(e)}")
            return []

    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    contents.append(text)
        except Exception as e:
            print(f"Failed to read TXT file: {file_path}\nError: {str(e)}")
            return []
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return contents

def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def embed_chunks(contents, filename):
    points = []
    absolute_chunk_index = 0

    for content_idx, content in enumerate(contents):
        for relative_idx, chunk in enumerate(chunk_text(content)):
            embedding = get_embedding(chunk)
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(f"Invalid embedding length: {len(embedding)}, expected {VECTOR_SIZE}")

            point = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "source": filename,
                    "content_id": f"{filename}_content_{content_idx+1}",
                    "chunk_index": relative_idx,
                    "absolute_index": absolute_chunk_index
                }
            }
            points.append(point)
            absolute_chunk_index += 1

    return points

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

def run_ingestion_pipeline():
    # Check if collection exists, if not create it
    if not collection_exists(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        create_collection(COLLECTION_NAME, VECTOR_SIZE)
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
    
    # Create index for source field (this is idempotent - won't fail if index already exists)
    print(f"Creating index for 'source' field...")
    try:
        create_source_index(COLLECTION_NAME)
        print("Index created successfully")
    except Exception as e:
        print(f"Index creation info: {e}")  # This might fail if index already exists, which is okay

    # Check if upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Folder '{UPLOAD_FOLDER}' not found.")
        return

    # Initialize Qdrant client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )

    # Get existing sources using the alternative method
    print("Checking for existing files...")
    existing_sources = get_all_sources_in_collection(client, COLLECTION_NAME)
    print(f"Found {len(existing_sources)} existing files in collection")

    # Process files in upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith((".pdf", ".docx", ".txt", ".json")):
            continue

        if filename in existing_sources:
            print(f"Skipping '{filename}' — already uploaded.\n")
            continue

        # Process and upload the file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing: {filename}")
        
        try:
            contents = load_text_from_file(file_path)
            if not contents:
                print(f"No content found in {filename}, skipping...")
                continue
                
            vectors = embed_chunks(contents, filename)
            response = upsert_vectors(COLLECTION_NAME, vectors)
            print("Qdrant upsert response:", response)
            print(f"Uploaded {len(vectors)} vectors for {filename}\n")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}\n")

# For the query asked by the user
def run_rag_pipeline(user_question):
    try:
        query_vector = embed_query(user_question)
        results = search_qdrant(query_vector)
        context = format_context(results)
        answer = ask_llama3(context, user_question)
        print("\n[DEBUG] Retrieved Context:\n", context)
        return answer
    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")
        return "I encountered an error while processing your question."

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