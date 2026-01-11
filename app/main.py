from fastapi import FastAPI, File, UploadFile
import uvicorn  
import shutil
from pathlib import Path
from db import load_vector_db, create_vectorDB
from rag import answer_question, load_and_chunk_pdf
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent   # project root
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Docu-Chat RAG")

@app.get('/')
def check():
    return {"status":"Hey Sir!"}

@app.post("/upload")
def upload_pdf(file:UploadFile = File(...)):
    file_path = UPLOAD_DIR/file.filename

    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

        chunks = load_and_chunk_pdf(str(file_path))

        create_vectorDB(chunks)

        return {
        "message": "PDF uploaded & indexed",
        "filename": file.filename
    }


# defining request schema
class QuestionRequest(BaseModel):
    question: str

@app.post('/ask')
def ask_question(payload:QuestionRequest):
    vectorstore = load_vector_db()
    answer, pages = answer_question(vectorstore, payload.question)
    return{
        "answer": answer,
        "pages": pages
    }
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)