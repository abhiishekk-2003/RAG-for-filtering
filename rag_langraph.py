# LangGraph CRAG implementation with nodes

import os
import json
import uuid
import requests
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, List
# Load environment variables
load_dotenv()

# Config
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

HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}", "Content-Type": "application/json"}
GROQ_HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
QDRANT_HEADERS = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}

# Main Functions

def get_embedding(text):
    payload = {"inputs": [text]}
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    if response.status_code != 200:
        print("Embedding Error:", response.text)
        response.raise_for_status()
    result = response.json()
    return result[0] if isinstance(result, list) else result

def embed_query(text):
    embedding = get_embedding(f"query: {text}")
    print("Query embedding ready. Vector size:", len(embedding))
    return embedding

def ask_llama3(context, question):
    prompt = f"""
You are a helpful assistant specialized in extracting doctor profiles based only on the provided context.
If the answer is not found in the context, respond strictly with: 'I cannot answer such questions.'

Return your answer in this structured JSON format:
{{
  \"Name\": \"\",
  \"Speciality\": \"\",
  \"Phone\": \"\",
  \"Address\": \"\",
  \"State\": \"\",
  \"City\": \"\",
  \"Education\": \"\",
  \"Experience\": \"\",
  \"Hospital\": \"\",
  \"Website\": \"\"
}}

Context:
{context}

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
    return response.json()["choices"][0]["message"]["content"].strip()

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
                                cleaned = item["content"].replace("\\n", " ").strip()

                                if cleaned:
                                    contents.append(cleaned)
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            contents.append(" ".join([page.extract_text() or "" for page in pdf.pages]))
    elif ext == ".docx":
        doc = Document(file_path)
        contents.append(" ".join([para.text for para in doc.paragraphs]))
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                contents.append(text)
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

def format_context(results):
    return "\n---\n".join([hit["payload"]["text"] for hit in results])

def search_qdrant(vector, top_k=5):
    payload = {
        "vector": vector,
        "top": top_k,
        "with_payload": True
    }
    response = requests.post(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search", headers=QDRANT_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["result"]

def upsert_vectors(collection_name, points):
    url = f"{QDRANT_URL}/collections/{collection_name}/points"
    payload = {"points": points}
    response = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    return response.json()

def create_collection(collection_name, vector_size):
    url = f"{QDRANT_URL}/collections/{collection_name}"
    payload = {"vectors": {"size": vector_size, "distance": "Cosine"}}
    response = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    return response.json()

# LangGraph Nodes

def node_embed_query(state):
    return {"query_vector": embed_query(state["question"])}

def node_search_qdrant(state):
    return {"results": search_qdrant(state["query_vector"])}

def node_format_context(state):
    return {"context": format_context(state["results"])}
    
def node_check_context_quality(state):
    prompt = f"Is the following context sufficient to answer this question: '{state['question']}'\nContext:\n{state['context']}\nAnswer only YES or NO."
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 32
    }
    response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload)
    verdict = response.json()["choices"][0]["message"]["content"].strip().lower()
    return "use_llm" if "yes" in verdict else "rephrase_query"

def node_rephrase_query(state):
    prompt = f"Rewrite the following question to retrieve more relevant chunks: {state['question']}"
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 64
    }
    response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload)
    new_question = response.json()["choices"][0]["message"]["content"].strip()
    return {"question": new_question}

def node_final_answer(state):
    return {"answer": ask_llama3(state["context"], state["question"])}

class GraphState(TypedDict):
    question: str
    query_vector: List[float]
    results: List[dict]
    context: str
    answer: str


# Build LangGraph

def build_crag_graph():
    builder = StateGraph(GraphState)
    builder.add_node("embed_query", RunnableLambda(node_embed_query))
    builder.add_node("search_qdrant", RunnableLambda(node_search_qdrant))
    builder.add_node("format_context", RunnableLambda(node_format_context))
    builder.add_node("rephrase_query", RunnableLambda(node_rephrase_query))
    builder.add_node("use_llm", RunnableLambda(node_final_answer))

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query", "search_qdrant")
    builder.add_edge("search_qdrant", "format_context")
    builder.add_conditional_edges("format_context", node_check_context_quality, {
        "use_llm": "use_llm",
        "rephrase_query": "rephrase_query"
    })
    builder.add_edge("rephrase_query", "embed_query")
    builder.set_finish_point("use_llm")
    return builder.compile()

# Ingestion and query
if __name__ == "__main__":
    rag_graph = build_crag_graph()
    mode = input("Choose mode: [1] Ingest Files  [2] Ask a Question: ").strip()

    if mode == "1":
        create_collection(COLLECTION_NAME, VECTOR_SIZE)
        for file in os.listdir(UPLOAD_FOLDER):
            if not file.endswith((".pdf", ".docx", ".txt", ".json")):
                continue

            # Skip file if already uploaded
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            existing = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=rest.Filter(
                    must=[rest.FieldCondition(
                        key="source",
                        match=rest.MatchValue(value=file)
                    )]
                ),
                limit=1
            )
            if existing and existing[0]:
                print(f"Skipping '{file}' — already uploaded.")
                continue
       
            contents = load_text_from_file(os.path.join(UPLOAD_FOLDER, file))
            vectors = embed_chunks(contents, file)
            upsert_vectors(COLLECTION_NAME, vectors)
            print("Uploaded", len(vectors), "vectors for", file)
    elif mode == "2":
        question = input("Ask your question: ")
        result = rag_graph.invoke({"question": question})
        print("\nStructured Answer:\n", result["answer"])
    else:
        print("Invalid input.")
