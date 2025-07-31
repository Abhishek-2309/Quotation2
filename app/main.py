from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from models.rag_service import index_pdf, search_image
import csv
import tempfile
from models.vl_model import load_model, run_answer
import tempfile, os, shutil
import re
import json
from fastapi import Query
from typing import List


app = FastAPI()

model, tokenizer = load_model()

Security_Service_queries = [
    {"key": "Company Name", "question": "Find the Company Name of the Security service, return in json with the key as 'Company Name' "},
    {"key": "Project", "question": """Identify the name of the project for which the security service or the guard is being provided. Do not confuse this with 'Bond Civil & Utility Construction' who are the company conducting this project. 
    The Project name may appear in the subject or the body of the letter addressed to the receiver/Document header and more
    It is the project for which the construction company would require security or guard services.
    Output should be in JSON format with the key as 'Project'. """},
    {"key": "Wage Type", "question": "Find the wage type of the guard as either 'Prevailing' or 'Non-Prevailing'. Check if either a prevailing or non-prevailing wage is mentioned within the document. If both are mentioned, write 'Prevailing/Non-Prevailing', if neither are mentioned explicitly leave empty, return in json with key as: 'Wage Type'"},
    {"key": "Year Quoted", "question": """Find the year in which this quotation or proposal was submitted or issued. 
    This year should be mentioned in the date of issuance, letter, proposal header, or signature area. 
    Do NOT return the year of founding, experience, or any certification expiry year. 
    Only return the year associated with the quotation document itself. 
    If not explicitly stated, leave it blank, return in json with key as: 'Year Quoted'"""}
]

rebar_queries = [
    {"key": "Company Name", "question": "Find the Company Name of the Rebar Providing service, return in json with the key as 'Company Name' "},
    {"key": "Epoxy Coated (Y/N)", "question": "Find whether rebar is epoxy coated or not(uncoated), return in json with the key as 'Epoxy Coated (Y/N)' and value as either Y/N"},
    {"key": "Scope Of Service", "question": "Find out the scope of service mentioned in the document, whether the company is willing to furnish or install or do both for rebar, return in json with key as: 'Scope Of Service'. Return a string: furnish (or) install (or) furnish & install"},
    {"key": "Average Unit Price ($/lb)", "question": """
    Find out the average unit price of installing the rebar in $/lb aka dollars per pounds as mentioned in the document, 
    Mention all the rebar types in the document and their price, write in terms of $/lb only, 
    If rebar weights are mentioned in terms of different units other than pounds, convert them to pounds and compute in terms of dollars per pound and give final value
    Write <price_1> in the format of '$ + <unit price value> + '/' + 'lb'
    If given separately say, total amount, return in json with key as: 'Average Unit Price ($/lb)'
    Finally give output as {
              "Average Unit Price ($/lb)": {
                "<Rebar Type 1>": <price_1>,
                "<Rebar Type 2>": <price_2>,
                ...
              }
            } in json"""},

    {"key": "Project", "question": "You are a document reader for Bond Civil & Utility Construction, Find out the name of the project for which the rebar is provided for, with key as 'Project'"},
    {"key": "Year Quoted", "question": """Find the year in which this quotation or proposal was submitted or issued. 
    This year should be mentioned in the date of issuance, letter, proposal header, or signature area. 
    Do NOT return the year of founding, experience, or any certification expiry year. 
    Only return the year associated with the quotation document itself. 
    If not explicitly stated, leave it blank, return in json with key as: 'Year Quoted' """}
    
]

firewall_queries = [
    {"key": "Company Name", "question": "Find the Company Name of the Firewall Providing service, return in json with the key as 'Company Name' "},
    {"key": "Project", "question": "You are a document reader for Bond Civil & Utility Construction, Find the name of the Project for which the company requires the firewall service, return in json with the key as 'Project"},
    {"key": "Year Quoted", "question": """Find the year in which this quotation or proposal was submitted or issued. 
    This year should be mentioned in the date of issuance, letter, proposal header, or signature area. 
    Do NOT return the year of founding, experience, or any certification expiry year. 
    Only return the year associated with the quotation document itself. 
    If not explicitly stated, leave it blank, return in json with key as: 'Year Quoted' """},
    {"key": "Total Length(LF)", "question": "What is the total linear footage (in LF) of the firewall? Look for summed wall segment lengths or configuration tables. If only panel lengths and quantities are given, calculate as sum of lengths for all the walls. Return only a single value in json with key as 'Total Length(LF)'"},
    {"key": "Average Height (LF)", "question": "What is the average height (in feet) of the firewall across all sections? Exmaine whether firewall heights are explicitly mentioned and give the average height of all firewalls. If not, search for height values in panel dimension formats (L×H×T where H is second), design notes, or wall specs. Find a single weighted average if multiple heights are given.Return only a single value in json with key as 'Average Height (LF)'"},
    {"key": "Total SF", "question": "What is the total square footage of the firewall? Look for surface area calculations, pricing tables, or summaries where it is mentioned in 'Sq.ft' or 'Square feet'. If unavailable, compute it as the: Total Length x Average Height. Ensure the result reflects full firewall coverage. Return only a single value in json with key as 'Total SF'"},
    {"key": "Total Price", "question": "Find the total cost of the firewall system only. Look for subtotal, lump sum, or extended price in quotations. Return as a number without currency symbols or commas in json with key as 'Total Price'"},
    {"key": "Average Unit Price ($/SF)", "question": "What is the average unit cost per square foot of the firewall? Look for $/SF entries or unit pricing in tables. If not directly shown, compute as (Total Price ÷ Total SF). Return a single value obtained after the extraction/calculation in json with key as 'Average Unit Price ($/SF)' "},
    {"key": "Width Range (in)", "question": "What is the thickness or width range (in inches) of the firewall panels? Look for panel specifications in drawings or text, especially in formats like L×H×T where thickness is the third value, or in callouts like ‘6” THK’. If not explicitly clear, give a single best estimate. Return a single value obtained after the estimation in json with key as 'Width Range (in)'"},
    {"key": "Hr-Rating", "question": "Find the Hr-Rating of the firewall which is the fire-resistance rating that indicates the duration, in hours, that the wall can withstand a standard fire test, return in json with the key as 'Hr-Rating"},
    
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

def process_sec_pdf(pdf_path: str) -> list[dict]:
    RAG = index_pdf(pdf_path)
    result_json = {}

    for q in Security_Service_queries:
        if q["key"] == "Unit Price ($/Hr)":
            continue
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

    # Wage Type Handling
    wage_type_info = result_json.get("Wage Type", "").strip()
    wage_types = []
    if wage_type_info:
        if "/" in wage_type_info:
            wage_types = [wt.strip() for wt in wage_type_info.split("/")]
        else:
            wage_types = [wage_type_info]

    if wage_types:
        wage_clauses = " ".join([
            f'Return the entry for "{wt}" under key "{wt}" in the output JSON.'
            for wt in wage_types
        ])
        wage_context = f"""
        In the below JSON chema, Return wages for the following type(s): {', '.join(wage_types)}.
        {wage_clauses}
        """
    else:
        wage_context = "There is no wage type, therefore in place of <Wage Type>, write ONLY 'None'. and fill the rest of the details as specified"

    # Unit Price Extraction
    final_unit_price_prompt = f"""
    Extract the Unit Price ($/Hr) without fail for the security guard from the provided image and stricttly fill up the JSON schema as per the following definitions:
    
    Identify the Unit Price ($/Hr) from the image. It can also be referred as Total Billing Rate/Total Guard Rate/Total Hourly Guard rate/Total Bill Rate
    - The *Unit Price* refers specifically to the *Total* hourly wage or billing rate of a security guard. It should be identified only in terms of $ per hr. 
    - The *Unit Price* is the total bill rate or the total guard rate after all taxes and benefits are considered. 
    - If multiple rates are given (e.g., base rate + additional/supplementary charges), compute the final effective hourly wage as the unit price.
    - Unit Price is not to be confused with Overtime or Holiday rates.

    Now fill up the JSON schema as follows:
    First, fill down the <Wage Type> based on the information below:
    {wage_context}
    Next, Identify the type/description of the security guard in place of <Type of Security Guard> if a distinct Unit Price/Billing Rate (defined below) is mentioned for the security guard.
    - If multiple types are found, ensure they all have distinct rates. Eg: Armed/Unarmed/Level 1
    - If no specific type is found, write the closest description from the image instead.
    
    Importantly, Identify if any Additional Rates are present.
    - Additional rates are separate from unit price/billing rate and should be listed separately. They are usually mentioned nearby the unit price and have higher rates than the unit price.
    - Additional rates can refer to - *Overtime* rate, *Holiday rate*, *Weekend rate*, if any such rate is present, find their value and write under Additional rates.
    - If overtime rates are mentioned in terms of billing rate/unit price, calculate and add its value.

    The JSON schema below is to be strictly followed:
    Format:
    {{
      "Unit Price ($/Hr)": {{
        "<Wage Type>": {{
          "<Type of Security Guard>": {{
            "Unit Price ($/Hr)": "$<final hourly wage>",
            "Additional rates": {{
              "<Overtime rate.>": "$<rate>",
              "<Holiday rate>": "$<rate>",
            }}
          }}
        }}
      }}
    }}
    """

    try:
        image = search_image(RAG, final_unit_price_prompt)
        result_text = run_answer(model, tokenizer, final_unit_price_prompt, image)
        extracted = extract_json(result_text)
        print(extracted)
        result_json["Unit Price ($/Hr)"] = extracted.get("Unit Price ($/Hr)", {})
    except Exception as e:
        result_json["Unit Price ($/Hr)"] = f"Error: {str(e)}"
    # Final Flattened Output: One row per guard type
    unit_price_data = result_json.get("Unit Price ($/Hr)", {})
    if not isinstance(unit_price_data, dict):
        return [{
            "Company Name": result_json.get("Company Name", ""),
            "Project": result_json.get("Project", ""),
            "Year Quoted": result_json.get("Year Quoted", ""),
            "Wage Type": "None",
            "Unit Price - Type": "",
            "Unit Price - Rate": "",
            "Unit Price - Additional Rates": "Invalid unit price format",
            "Notes": "Error"
        }]

    output_rows = []

    def parse_additional_rates(rates_dict):
        if not isinstance(rates_dict, dict):
            return ""
        return ", ".join([f"{k}: {v}" for k, v in rates_dict.items()])

    if wage_types:
        for wt in wage_types:
            guard_entries = unit_price_data.get(wt, {})
            for guard_type, data in guard_entries.items():
                output_rows.append({
                    "Company Name": result_json.get("Company Name", ""),
                    "Project": result_json.get("Project", ""),
                    "Year Quoted": result_json.get("Year Quoted", ""),
                    "Wage Type": wt,
                    "Unit Price - Type": guard_type,
                    "Unit Price - Rate": data.get("Unit Price ($/Hr)", ""),
                    "Unit Price - Additional Rates": parse_additional_rates(data.get("Additional rates", {})),
                    "Notes": result_json.get("Notes", "")
                })
    else:
        guard_entries = unit_price_data.get("None", unit_price_data)
        for guard_type, data in guard_entries.items():
            output_rows.append({
                "Company Name": result_json.get("Company Name", ""),
                "Project": result_json.get("Project", ""),
                "Year Quoted": result_json.get("Year Quoted", ""),
                "Wage Type": "None",
                "Unit Price - Type": guard_type,
                "Unit Price - Rate": data.get("Unit Price ($/Hr)", ""),
                "Unit Price - Additional Rates": parse_additional_rates(data.get("Additional rates", {})),
                "Notes": result_json.get("Notes", "")
            })

    return output_rows

def process_rebar_pdf(pdf_path: str) -> dict:
    RAG = index_pdf(pdf_path)
    result_json = {}

    for q in rebar_queries:
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

    rebar = result_json.get("Average Unit Price ($/lb)", {})

    result_lst = []
    for rebar_type, price in rebar.items():
        result_lst.append({
                    "Company Name": result_json.get("Company Name", ""),
                    "Epoxy Coated (Y/N)": result_json.get("Epoxy Coated (Y/N)", ""),
                    "Scope Of Service": result_json.get("Scope Of Service", ""),
                    "Type": rebar_type,
                    "Price": price,
                    "Project": result_json.get("Project", ""),
                    "Year Quoted": result_json.get("Year Quoted", ""),
                    "Notes": result_json.get("Notes", "")
                })

    return result_lst

def process_firewall_pdf(pdf_path: str) -> dict:
    RAG = index_pdf(pdf_path)
    result_json = {}
    for q in firewall_queries:
        print(q)
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
    
    return [result_json]





















#endpoints

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

@app.post("/extract-folder")
async def upload_and_process_folder(zip_file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    
    try:
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
                        print(results)
                        all_results.extend(results)
                    except Exception as e:
                        all_results.append({
                            "PDF": fname,
                            "Error": str(e)
                        })
    
        if not all_results:
            raise HTTPException(status_code=404, detail="No PDF files processed")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


    # Write to CSV
    csv_path = os.path.join(tempfile.gettempdir(), "batch_extraction_results.csv")
    with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    return FileResponse(csv_path, filename="extracted_results.csv", media_type="text/csv")
