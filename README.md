# 📚 RAG Pipeline for File-Based Question Answering

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows you to upload documents (PDF, DOCX, TXT, JSON), convert them into embeddings using a Groq-powered LLaMA3 model, and store them in a **Qdrant vector database**. You can then query these documents using natural language questions.

---

## 🚀 Features

- 🧾 Upload multiple file formats (`.pdf`, `.docx`, `.txt`, `.json`)
- 🔍 Text chunking and semantic embedding (using `thenlper/gte-small`)
- 🧠 Embedding storage and retrieval via **Qdrant Cloud**
- 💬 Query interface that fetches relevant chunks and uses **LLaMA3 (via Groq)** to generate answers
- 🧰 Designed for local usage and easy UI integration

---

## 📁 Folder Structure

RAG-for-filtering/

├── main.py

├── upload_here/              # Folder to drop files for ingestion

├── utils/

│   ├── embedder.py           # Embedding logic

│   ├── qdrant_utils.py       # Vector DB creation and upsert

│   ├── embed_query.py        # Query embedding

│   ├── retriever.py          # Qdrant search

│   └── groq_llm.py           # Call LLaMA3 model

```
---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/RAG-for-filtering.git
cd RAG-for-filtering
```
### 2. Install delendencies:
```
pip install -r requirements.txt
```
### 3. Add Your .env File:
reate a .env file in the project root with the following values:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-qdrant-instance.cloud
COLLECTION_NAME=your_collection_name
```
### 📥 Ingest Files
Place your .pdf, .docx, .txt, or .json files inside the upload_here/ folder.
Then run:
```
python main.py
```
Choose option [1] Ingest Files when prompted.
❓ Ask Questions
To ask questions from the ingested documents:
```
python main.py
```
Choose option [2] Ask a Question and type your query.

Example:
```
Choose mode: [1] Ingest Files  [2] Ask a Question: 2
Ask a question: What are the key findings of the uploaded research paper?
```

## 🧪 Supported File Formats
-	PDF – Text extracted using pdfplumber
-	DOCX – Paragraphs extracted with python-docx
-	TXT – Plain text
-	JSON – Custom support for Bing/Google-style webSearchResults content

## 📌 Coming Soon
- ✅ Web-based UI using Streamlit or FastAPI
- ✅ Source highlighting in answers
- ✅ Filter by document or metadata
 
