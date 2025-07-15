from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models.rag_service import index_pdf, search_image
from models.vl_model import load_model, run_answer
import tempfile, os
import re
import json

app = FastAPI()

model, tokenizer = load_model()

queries = [
    {
        "key": "Unit Price ($/Hr)",
        "question": """Find the Unit Price($/Hr) per Hour of all the types of security guards mentioned. The Unit Price($/Hr) refers to the Wages of the Security Guard per hour. 
The unit price should be the the total wage rate obtained after all the conditional charges.
The output should be a json schema with the key consisting of the type(s) of security guard mentioned and the value being all the conditional charges alongside the base rate. The Unit Price($/hr) should be the total after all these are considered. 
If overtime rates or any other rates are mentioned, they should be a separate value under the same type of security guard
If multiple Categories of guards/Wages are mentioned, mention them as separate keys within the json schema.
The final output should be a json schema with the key being 'Unit Price($/Hr)' and its value being all these previously obtained security guards.
Note: If both prevailing/non prevailing rates are mentioned, create separate JSON entries for both following the previous schema.
"""
    },
    {"key": "Company Name", "question": "Find the Company Name of the Security service, return in json with the key as 'Company Name' "},
    {"key": "Project", "question": "Find the name of the Project for which the security service is provided, return in json with the key as 'Project"},
    {"key": "Wage Type", "question": "Find the wage type of the guard as either prevailing or non prevailing. If both are mentioned, write prevailing/non prevailing, if neither are mentioned explicitly leave empty, return in json with key as: 'Wage Type'"},
    {"key": "Year Quoted", "question": "Find the year of the project mentioned in the quotation document, note this is the year in which the quotation is submitted, if not explicitly mentioned leave empty, return in json with key as: 'Year Quoted'"}
]

def extract_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(1).strip()
    else:
        # fallback: try finding any JSON-looking block
        fallback = re.search(r"(\{.*?\})", text, re.DOTALL)
        if fallback:
            try:
                return json.loads(fallback.group(1))
            except json.JSONDecodeError:
                return fallback.group(1).strip()
        return text.strip()  # return raw text if nothing found


@app.post("/extract")
async def extract_from_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    RAG = index_pdf(tmp_file_path)  

    result_json = {}
    for q in queries:
        try:
            image = search_image(RAG, q["question"])  # Always uses latest doc
            result_text = run_answer(model, tokenizer, q["question"], image)
            print(result_text)
            result_json[q["key"]] = extract_json(result_text)[q['key']]
        except Exception as e:
            result_json[q["key"]] = f"Error: {str(e)}"

    os.unlink(tmp_file_path)
    return JSONResponse(content=result_json)
