#!/usr/bin/env python

# Test script to run Gemma 3.0 27B
# Downloads the model weights if not already on disk.

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

url1 = "https://s3-eu-west-1.amazonaws.com/textract.samples/Farragut-DD-348-1942-01-0021.jpg"

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {
                "type": "text",
                "text": "Give the ship's Latitude and Longitude at Noon.",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
