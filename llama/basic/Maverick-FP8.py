#!/usr/bin/env python

# Test script to run Llama Maverick FP8
# Downloading the weights if not already on disc.

import os

from huggingface_hub import login

login(token=os.getenv("HF_KEY"))

from transformers import (
    AutoProcessor,
    Llama4ForConditionalGeneration,
    FbgemmFp8Config,
)
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"


processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=FbgemmFp8Config(),
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {
                "type": "text",
                "text": "Can you describe how these two images are similar, and how they differ?",
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
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[0]
print(response)
print(outputs[0])
