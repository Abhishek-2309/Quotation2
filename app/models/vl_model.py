from unsloth import FastVisionModel
from transformers import TextStreamer
import torch

def load_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

def run_answer(model, tokenizer, question: str, image):
    prompt = f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}"""

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512, use_cache=True)
    return tokenizer.decode(_, skip_special_tokens=True)
