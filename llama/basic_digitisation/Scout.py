#!/usr/bin/env python

# Test Llama Scout on a simple digitisation task

import os

from huggingface_hub import login

login(token=os.getenv("HF_KEY"))

from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

url1 = "https://s3-eu-west-1.amazonaws.com/textract.samples/Farragut-DD-348-1942-01-0021.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {
                "type": "text",
                "text": "This page contains three sets of latitude and longitude coordinates. "
                + "Please extract all three sets and return them in a list format.",
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
