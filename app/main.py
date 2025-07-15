from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from models.rag_service import index_pdf, search_image
from models.vl_model import load_model, run_answer
import tempfile, os
import re
import json

app = FastAPI()
router = APIRouter()
app.include_router(router)

model, tokenizer = load_model()

queries = [
    {
        "key": "Unit Price ($/Hr)",
        "question": """Extract the Unit Price ($/Hr) for all types of security guards mentioned in this quotation. The Unit Price is the final hourly wage after considering all additional or conditional charges.
Each type of guard should be a key in a nested JSON, and its value should include:
- Base Wages which is the wage before adding the additional charges to make up the Unit Price, if not explicitly stated, ignore.
- Any conditional/additional charges (e.g., allowances, taxes, fees)
- The final computed Unit Price ($/Hr) after all additions
- If overtime or weekend rates are mentioned, include them under separate keys inside the same guard type
Note:
If the given guard type is explicitly shown to be Prevailing/Non Prevailing, Classify the type as either Prevailing or Non-Prevailing. If neither is present classify it as 'None', do NOT assume wage type as either. If both Prevailing and Non-Prevailing wages are mentioned, separate them into two distinct JSON objects under keys `"Prevailing"` and `"Non-Prevailing"`.
Guard Type Should be explicitly clear. It is defined for a SINGLE guard only. Do not assume any additional guards or security as guard type, each guard type should clearly have hourly wages. All wages are in $/hr, Do not include any other wages not defined in this format.

Return the final output in the following JSON structure:
```json
{
  "Unit Price ($/Hr)": {
    "<Prevailing/Non-Prevailing/None>": {
      "Security Guard Type A": {
        "Base Wage": "$X",
        "Additional Charges": {
          "<field1 from document>": "$...",
          "<field1 from document>": "$..."
        },
        "Unit Price ($/Hr)": "$Total"
        "<Any special rates for overtime/holiday in document>": "$..."
      },
      ...
    },
    "<Prevailing/Non-Prevailing/None>": {
      ...
    }
  }
}
"""
    },
    {"key": "Company Name", "question": "Find the Company Name of the Security service, return in json with the key as 'Company Name' "},
    {"key": "Project", "question": "Find the name of the Project for which the security service is provided, return in json with the key as 'Project"},
    {"key": "Wage Type", "question": "Find the wage type of the guard as either 'Prevailing' or 'Non-Prevailing'. If both are mentioned, write 'Prevailing/Non-Prevailing', if neither are mentioned explicitly leave empty, return in json with key as: 'Wage Type'"},
    {"key": "Year Quoted", "question": """Find the year in which this quotation or proposal was submitted or issued. 
    This year should be mentioned in the date of issuance, letter, proposal header, or signature area. 
    Do NOT return the year of founding, experience, or any certification expiry year. 
    Only return the year associated with the quotation document itself. 
    If not explicitly stated, leave it blank, return in json with key as: 'Year Quoted'"""}
]

def strip_prompt_from_output(text: str) -> str:
    """
    Removes everything before and including the last occurrence of 'assistant\n' in the model output.
    Assumes that the JSON starts right after this.
    """
    split_pattern = r"(?:^|\n)assistant\s*\n"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()


def extract_json(text: str):
    """
    Strips prompt using 'strip_prompt_from_output' and tries to load clean JSON.
    Also unwraps markdown fences like ```json ... ``` if needed.
    """
    text = strip_prompt_from_output(text)

    # Remove markdown-style ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # fallback: extract the first JSON-like block
        fallback = re.search(r"(\{.*\})", text, re.DOTALL)
        json_str = fallback.group(1).strip() if fallback else text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return json_str  # return raw string if still unparseable

def process_pdf(pdf_path: str) -> dict:
    RAG = index_pdf(pdf_path)
    result_json = {}

    for q in queries:
        try:
            image = search_image(RAG, q["question"])
            result_text = run_answer(model, tokenizer, q["question"], image)
            extracted = extract_json(result_text)
            if isinstance(extracted, dict) and q["key"] in extracted:
                val = extracted[q["key"]]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                result_json[q["key"]] = val
            else:
                result_json[q["key"]] = extracted
        except Exception as e:
            result_json[q["key"]] = f"Error: {str(e)}"
    
    unit_price_data = result_json.get("Unit Price ($/Hr)", {})
    company_name = result_json.get("Company Name", "")
    project = result_json.get("Project", "")
    year_quoted = result_json.get("Year Quoted", "")
    wage_type_fallback = result_json.get("Wage Type", "")

    def create_final_obj(wage_type_key: str, price_dict: dict):
        return {
            "PDF": os.path.basename(pdf_path),
            "Unit Price ($/Hr)": price_dict,
            "Company Name": company_name,
            "Project": project,
            "Wage Type": wage_type_key,
            "Year Quoted": year_quoted
        }

    final_outputs = []

    if wage_type_fallback:
        if isinstance(unit_price_data, dict):
            if "Prevailing" in unit_price_data:
                final_outputs.append(create_final_obj("Prevailing", unit_price_data["Prevailing"]))
            if "Non-Prevailing" in unit_price_data:
                final_outputs.append(create_final_obj("Non-Prevailing", unit_price_data["Non-Prevailing"]))
    else:
        final_outputs.append(create_final_obj(wage_type_fallback, unit_price_data.get("None", {})))

    return final_outputs



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
            extracted = extract_json(result_text)
            if isinstance(extracted, dict) and q["key"] in extracted:
                val = extracted[q["key"]]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                result_json[q["key"]] = val
            else:
                result_json[q["key"]] = extracted
    
        except Exception as e:
            result_json[q["key"]] = f"Error: {str(e)}"
    
    print(result_json)
    os.unlink(tmp_file_path)
    unit_price_data = result_json.get("Unit Price ($/Hr)", {})
    company_name = result_json.get("Company Name", "")
    project = result_json.get("Project", "")
    year_quoted = result_json.get("Year Quoted", "")
    wage_type_fallback = result_json.get("Wage Type", "")

    def create_final_obj(wage_type_key: str, price_dict: dict):
        return {
            "Unit Price ($/Hr)": price_dict,
            "Company Name": company_name,
            "Project": project,
            "Wage Type": wage_type_key,
            "Year Quoted": year_quoted
        }

    final_outputs = []

    if wage_type_fallback:
        if isinstance(unit_price_data, dict):
            has_prevailing = "Prevailing" in unit_price_data
            has_non_prevailing = "Non-Prevailing" in unit_price_data
    
            if has_prevailing:
                final_outputs.append(create_final_obj("Prevailing", unit_price_data["Prevailing"]))
            if has_non_prevailing:
                final_outputs.append(create_final_obj("Non-Prevailing", unit_price_data["Non-Prevailing"]))
    else:
        # If the model returned the unit price as a string or non-dict
        final_outputs.append(create_final_obj(wage_type_fallback, unit_price_data["None"]))

    return JSONResponse(content=final_outputs)

from fastapi import Query
from typing import List
@router.post("/extract-folder")
async def upload_and_process_folder(zip_file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, zip_file.filename)
        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())

        shutil.unpack_archive(zip_path, tmp_dir)

    all_results = []

    for root, _, files in os.walk(tmp_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, fname)
                try:
                    results = process_pdf(pdf_path)
                    all_results.extend(results)
                except Exception as e:
                    all_results.append({
                        "PDF": fname,
                        "Error": str(e)
                    })

    if not all_results:
        raise HTTPException(status_code=404, detail="No PDF files processed")

    # Write to CSV
    csv_path = os.path.join(tempfile.gettempdir(), "batch_extraction_results.csv")
    with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    return FileResponse(csv_path, filename="extracted_results.csv", media_type="text/csv")
