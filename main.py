# main.py
# For extracting text from uploaded files
import os
import pdfplumber
from utils.embedder import get_embedding
from utils.qdrant_utils import upsert_vectors, create_collection
import uuid
from docx import Document
import os
from dotenv import load_dotenv

load_dotenv()

# For retrieveing and asking a question
from utils.embed_query import embed_query
from utils.retriever import search_qdrant
from utils.formatter import format_context
from utils.groq_llm import ask_llama3
from qdrant_client.http import models as rest

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_here")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_SIZE = 384  # GTE-small outputs 384-d vectors

def load_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".json":
        import json
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            contents = []
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
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            return [text] if text.strip() else []
    else:
        raise ValueError(f"Unsupported file type: {ext}")    

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
            if len(embedding) != 384:
                raise ValueError("Invalid embedding length")

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

def run_ingestion_pipeline():
    #Creating collection
    create_collection(COLLECTION_NAME, VECTOR_SIZE)

    #Uploading all files in folder
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Folder '{UPLOAD_FOLDER}' not found.")
        return

    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith((".pdf", ".docx", ".txt", ".json")):
            continue

        #Checking if files is already in DB
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )

        #Finding at least one point with this source
        existing = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="source",
                        match=rest.MatchValue(value=filename)
                    )
                ]
            ),
            limit=1
        )

        if existing and existing[0]:
            print(f" Skipping '{filename}' — already uploaded.\n")
            continue

        #Files to split and store
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f" Processing: {filename}")
        contents = load_text_from_file(file_path)
        vectors = embed_chunks(contents, filename)
        
        response = upsert_vectors(COLLECTION_NAME, vectors)
        print("Qdrant upsert response:", response)
        print(" Uploaded", len(vectors), "vectors for", filename, "\n")

# For the query asked by the user
def run_rag_pipeline(user_question):
    query_vector = embed_query(user_question)
    results = search_qdrant(query_vector)
    context = format_context(results)
    answer = ask_llama3(context, user_question)
    print("\n[DEBUG] Retrieved Context:\n", context)
    return answer

if __name__ == "__main__":
    mode = input("Choose mode: [1] Ingest Files  [2] Ask a Question: ").strip()

    if mode == "1":
        run_ingestion_pipeline()
    elif mode == "2":
        question = input("Ask a question: ")
        answer = run_rag_pipeline(question)
        print("\n Answer:\n", answer)
    else:
        print(" Invalid option selected.")                                                                                                                 