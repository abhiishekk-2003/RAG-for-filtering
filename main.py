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
    text = ""

    if ext == ".pdf":
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

    elif ext == ".json":
        import json
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            lines = []
            if isinstance(data, dict) and "webSearchResults" in data:
                for entry in data["webSearchResults"]:
                    if isinstance(entry, list):
                        for item in entry:
                            if isinstance(item, dict) and "content" in item:
                                from bs4 import BeautifulSoup
                                import re

                                raw = item["content"]
                                # Basic de-noising
                                cleaned = raw.replace("\n", " ").replace("  ", " ").strip()

                                # Optionally remove repeated words or patterns
                                cleaned = re.sub(r"\b(\w+)( \1\b)+", r"\1", cleaned)  # removes repeated words
                                cleaned = re.sub(r"[^\w\s,.:-]", "", cleaned)         # remove stray characters

                                # Optional: If there's HTML in content
                                # cleaned = BeautifulSoup(raw, "html.parser").get_text()

                                lines.append(cleaned)


            text = "\n".join(lines)

        except Exception as e:
            print(f"Failed to read JSON file: {file_path}")
            print("Error:", str(e))
            text = ""  # fallback to empty so your pipeline continues

    return text



def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def embed_chunks(chunks, filename):
    points = []
    for chunk in chunks:
        embedding = get_embedding(chunk)

        # HF API wraps the embedding in another list — extract the vector
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]

        #Validating length
        if len(embedding) != 384:
            raise ValueError(f"Invalid embedding length: {len(embedding)}")

        point = {
            "id": str(uuid.uuid4()),  # Use UUID, not file name ID as Qdrant not supports that format
            "vector": embedding,
            "payload": {
                "text": chunk,
                "source": filename
            }

        }

        points.append(point)

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
        text = load_text_from_file(file_path)
        chunks = list(chunk_text(text))
        vectors = embed_chunks(chunks, filename)
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
    mode = input("Choose mode: [1] Ingest PDFs  [2] Ask a Question: ").strip()

    if mode == "1":
        run_ingestion_pipeline()
    elif mode == "2":
        question = input("Ask a question: ")
        answer = run_rag_pipeline(question)
        print("\n Answer:\n", answer)
    else:
        print(" Invalid option selected.")                                                                                                                 