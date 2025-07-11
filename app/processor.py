from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from pdf2image import convert_from_path
import torch
import os
import uuid

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL")

def qwen_extract_fields(image_path, doc_prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=doc_prompt, images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

def process_pdf_to_images(pdf_path, dpi=200):
    output_paths = []
    images = convert_from_path(pdf_path, dpi=dpi)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join("uploads", f"{base_name}_pages")
    os.makedirs(output_dir, exist_ok=True)
    for i, page in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        page.save(image_path, "PNG")
        output_paths.append(image_path)
    return output_paths
