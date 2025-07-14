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

from transformers import TextStreamer
from unsloth import FastVisionModel
import torch

def run_answer(model, tokenizer, question: str, image):
    prompt = f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}"""

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # Tokenize text only
    text_inputs = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    # Process image separately
    vision_inputs = model.process_images([image])  # returns pixel values
    vision_inputs = vision_inputs.to("cuda")

    # Run generation
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **text_inputs,
        images=vision_inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
