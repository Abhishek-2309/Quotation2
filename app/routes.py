from fastapi import APIRouter, UploadFile, File
import os
from app.processor import qwen_extract_fields
from app.utils import build_prompt_from_doc

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/extract-fields/")
async def extract_fields_from_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Use Qwen to directly process image and extract content
    doc_text = "Document content is in the image itself."
    prompt = build_prompt_from_doc(doc_text)

    response = qwen_extract_fields(file_path, prompt)
    return {"filename": file.filename, "extracted_fields": response}