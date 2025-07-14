from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models.rag_service import index_pdf, search_image
from models.vl_model import load_model, run_answer
import tempfile, os

app = FastAPI()

model, tokenizer = load_model()

queries = [
    {
        "key": "Unit Price Table",
        "question": """Find the Unit Price($/Hr) per Hour which refers to the Wages of the Security Guard per hour. 
If multiple Categories of guards are mentioned, mention them as separate rows in a table. 
If same category of guard has multiple subcategories, mention them as subrows within main row.
Each conditional charge alongside the base rate is a column. Unit Price should be the total after all these are considered.
After getting the unit price table, write it as a json entry within the schema with keys representing columns and value representing corresponding entry"""
    },
    {"key": "Company Name", "question": "Find the Company Name, return in json"},
    {"key": "Project Name", "question": "Find the name of the Project, return in json"},
    {"key": "Wage Type", "question": "Find the wage type as either prevailing or non prevailing if not mentioned explicitly leave empty, return in json"},
    {"key": "Year", "question": "Find the year of the project, return in json"}
]

@app.post("/extract")
async def extract_from_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    index_pdf(tmp_file_path)

    result_json = {}
    for q in queries:
        try:
            image = search_image(q["question"])
            result_text = run_answer(model, tokenizer, q["question"], image)
            result_json[q["key"]] = result_text
        except Exception as e:
            result_json[q["key"]] = f"Error: {str(e)}"

    os.unlink(tmp_file_path)
    return JSONResponse(content=result_json)
