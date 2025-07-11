from colpali import CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("colpali/clip-vit-base-patch32")
model.eval()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().flatten()
    return embedding


def get_text_embedding(text):
    with torch.no_grad():
        embedding = model.encode_text(text).cpu().numpy().flatten()
    return embedding


# document_extractor/app/models/qwen_infer.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL")

def run_qwen_vl(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
