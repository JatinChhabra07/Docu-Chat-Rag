from fastapi import FastAPI, File, UploadFile
import uvicorn  
import shutil
from pathlib import Path

UPLOAD_DIR = Path("data/uploads")
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

        return {
        "message": "File uploaded successfully",
        "filename": file.filename
    }
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)