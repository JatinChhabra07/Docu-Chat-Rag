# Clean RAG: Chat with Your Documents 📄

A production-style Retrieval-Augmented Generation (RAG) system
that allows users to upload PDFs and ask questions with
source page numbers.

## Features
- PDF upload via FastAPI
- Chunking with metadata preservation
- Vector search using FAISS
- Answers with page number citations
- Clean project structure

## Tech Stack
- FastAPI
- LangChain
- FAISS
- Ollama Embeddings

## How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
