#from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
#from fastapi.responses import JSONResponse, FileResponse
from models.rag_service import index_pdf, search_image
from PIL import Image
#import csv
import tempfile
from models.vl_model import load_model, run_answer
import tempfile, os, shutil
import re
import json
#from fastapi import Query
from typing import List

"""
app = FastAPI()
"""

model, tokenizer = load_model()

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

def is_valid(value):
    try:
        if isinstance(value, str):
            value = value.strip()
            if value.lower().startswith("error") or value == "":
                return False
        float_val = float(value)
        return float_val > 0
    except:
        return False

def clean_numeric(value):
    """Removes currency, commas, etc."""
    if isinstance(value, str):
        return value.replace('$', '').replace(',', '').strip()
    return value


#--------------------------------INVOICE-----------------------------------------
Invoice_queries = [
    {"key": "Invoice Number", "question": "Find the Invoice Number in the image. Return in json with the key as 'Invoice Number' "},
    
    {"key": "Invoice Date", "question": "Find the Invoice Date in the image. Return in json with the key as 'Invoice Date' "},
    
    {"key": "Buyer's Information", "question": """Identify the Buyer and extract the following buyer details:
    Name(either Buyer or Company), Address, Contact, GSTIN(GSTIN Number of Buyer's company)
    Output should be in JSON format with the key as 'Buyer's Information' and values as the above details. """},
    
    {"key": "Seller's Information", "question": """Identify the Seller and extract the following seller details:
    Name(either seller or Company), Address, Contact, GSTIN(GSTIN Number of Seller's company)
    Output should be in JSON format with the key as 'Seller's Information' and values as the above details. """},
    
    {"key": "Main Table", "question": """Identify the main line table from the image containing the line items of the Invoice document. This is the itemized list of products for which Invoice is performed
    Only add all the items present in that table. Do Not include totals and others.
    Return in a strict JSON format only. The value of the main key should contain each row's fields and corresponding value for all the rows in the main table.
    Output should be in JSON format with the key as 'Items' and values as each row of fields and values """},
    
    {"key": "Payment Terms", "question": """Identify the following Payment terms from the given image:
    Bank_details, consisting of: Bank_Name, IFSC_Code, Bank_account_no
    Payment Due Date,
    Payment Methods
    Output should be in JSON format with the key as 'Payment Terms' and values as the above details """},
    
    {"key": "Summary", "question": """Identify The following summary details:
    Subtotal(Total amount of goods before taxes), Taxes, Discounts, Total_Amount_Due(Total amount due including Taxes)
    Output should be in JSON format with the key as 'Summary' and values as the above details. """},   
    
    {"key": "Other_Important_Sections", "question": """Identify The following details:
    Terms and conditions, Notes/Comments, Signature.
    Output should be in JSON format with the key as 'Other_Important_Sections' and values as the above details. """},   
]

def Process_Invoice(path: str) -> dict:
    is_pdf = False
    if path.lower().endswith('pdf'):
        is_pdf = True
        RAG = index_pdf(path)
    else:
        img = Image.open(path).convert("RGB")
        
    result_json = {}

    for q in Invoice_queries:
        try:
            if is_pdf:
                image = search_image(RAG, q["question"])
                result_text = run_answer(model, tokenizer, q["question"], image)
            else:
                result_text = run_answer(model, tokenizer, q["question"], img)
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

    return result_json

