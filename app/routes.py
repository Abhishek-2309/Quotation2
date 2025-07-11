from fastapi import APIRouter, UploadFile, File
import os
from app.processor import qwen_extract_fields, process_pdf_to_images
from app.utils import build_prompt_from_doc

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/extract-fields/")
async def extract_fields_from_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.lower().endswith(".pdf"):
        image_paths = process_pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    # Use Qwen to directly process image(s)
    doc_text = "Document content is in the image itself."
    prompt = build_prompt_from_doc(doc_text)

    all_outputs = []
    for image_path in image_paths:
        response = qwen_extract_fields(image_path, prompt)
        all_outputs.append({"image": os.path.basename(image_path), "fields": response})

    return {"filename": file.filename, "results": all_outputs}
