#!/usr/bin/env python

# Get Gemma to transcribe data from a daily rainfall sheet

from utils.hf import HFlogin

HFlogin()

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# Test image to get data from
# url1 = "https://brohan.org/AI_daily_precip/_images/Devon_1941-1950_RainNos_1651-1689-293.jpg"
# url1 = "https://brohan.org/AI_daily_precip/_images/missing_infilled.jpg"
url1 = "https://brohan.org/AI_daily_precip/_images/original.jpg"
response = requests.get(url1)
image = Image.open(BytesIO(response.content))

# Crop and resize the image so it's 850 pixels square
# I.e. so that it's the optimum shape for Gemma.
# I don't want to distort it, so I will crop it to a square.
crop = image.crop([0, image.size[1] - image.size[0], image.size[0], image.size[1]])
smaller = crop.resize([850, 850])

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The pages you are working on are records of daily rainfall from the UK Met Office. "
    + "Each page contains part of a data table - the first column contains the day of the month, "
    + "the second column contains the rainfall value for that day in January, "
    + "the third column contains the rainfall value for that day in February, "
    + "and so on for each month of the year. "
)

Questions = [
    "List the rainfall values for the 10th to 20th of January. ",
    "List the rainfall values for the 10th to 20th of February. ",
    # "List the rainfall values for each day in March. ",
    # "List the rainfall values for each day in April. ",
    # "List the rainfall values for each day in May. ",
    # "List the rainfall values for each day in June. ",
    # "List the rainfall values for each day in July. ",
    # "List the rainfall values for each day in August. ",
    # "List the rainfall values for each day in September. ",
    # "List the rainfall values for each day in October. ",
    # "List the rainfall values for each day in November. ",
    # "List the rainfall values for each day in December. ",
]

for q in Questions:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": s_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": smaller},
                {
                    "type": "text",
                    "text": q,
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
        generation = model.generate(
            **inputs, max_new_tokens=1000, do_sample=False, top_k=None, top_p=None
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
