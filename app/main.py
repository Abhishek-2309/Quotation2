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
        "question": """Extract the Unit Price ($/Hr) for all types of security guards mentioned in this quotation. The Unit Price is the final hourly wage after considering all additional or conditional charges.

Each type of guard should be a key in a nested JSON, and its value should include:
- Base Wages which is the wage before adding the additional charges to make up the Unit Price, if not explicitly stated, ignore.
- Any conditional/additional charges (e.g., allowances, taxes, fees)
- The final computed Unit Price ($/Hr) after all additions
- If overtime or weekend rates are mentioned, include them under separate keys inside the same guard type
Do not return shift descriptors or schedule labels as keys unless they are clearly labeled as guard types. Only If explicitly mentioned that wages are Prevailing/Non Prevailing, classify them. If no prevailing/non-prevailing split is found, just include a single dictionary under "Unit Price ($/Hr)".
If both Prevailing and Non-Prevailing wages are mentioned, separate them into two distinct JSON objects under keys `"Prevailing"` and `"Non-Prevailing"`.

Return the final output in the following JSON structure:
```json
{
  "Unit Price ($/Hr)": {
    "Prevailing": {
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
    "Non-Prevailing": {
      ...
    }
  }
}
"""
    },
    {"key": "Company Name", "question": "Find the Company Name of the Security service, return in json with the key as 'Company Name' "},
    {"key": "Project", "question": "Find the name of the Project for which the security service is provided, return in json with the key as 'Project"},
    {"key": "Wage Type", "question": "Find the wage type of the guard as either 'Prevailing' or 'Non-Prevailing'. If both are mentioned, write 'Prevailing/Non-Prevailing', if neither are mentioned explicitly leave empty, return in json with key as: 'Wage Type'"},
    {"key": "Year Quoted", "question": "Find the year quoted as mentioned in the quotation document, note this is the year associated with the document's creation or issuance, if not explicitly mentioned leave empty, return in json with key as: 'Year Quoted'"}
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

    if isinstance(unit_price_data, dict):
        has_prevailing = "Prevailing" in unit_price_data
        has_non_prevailing = "Non-Prevailing" in unit_price_data

        if has_prevailing:
            final_outputs.append(create_final_obj("Prevailing", unit_price_data["Prevailing"]))
        if has_non_prevailing:
            final_outputs.append(create_final_obj("Non-Prevailing", unit_price_data["Non-Prevailing"]))
        if not has_prevailing and not has_non_prevailing:
            # No separate categories, fallback
            final_outputs.append(create_final_obj(wage_type_fallback, unit_price_data))
    else:
        # If the model returned the unit price as a string or non-dict
        final_outputs.append(create_final_obj(wage_type_fallback, unit_price_data))

    return JSONResponse(content=final_outputs)
