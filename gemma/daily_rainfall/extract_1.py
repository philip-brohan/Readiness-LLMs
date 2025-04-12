#!/usr/bin/env python

# Get Gemma to transcribe data from a daily rainfall sheet

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# Test image to get data from
# url1 = "https://brohan.org/AI_daily_precip/_images/Devon_1941-1950_RainNos_1651-1689-293.jpg"
url1 = "https://brohan.org/AI_daily_precip/_images/missing_infilled.jpg"

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The pages you are working on are records of daily rainfall from the UK Met Office. "
    + "Each page contains the data from one weather station for one year. One entry for each day in the year."
    + "The values are in a table where the columns are the months, and the rows are the days in each month. "
    + "The entries are rainfall values. On some days it does not rain, and the entry for that day may be blank or contain a dash. "
    + "When asked for the value for that day, you should return '-' if there is no number in the corresponding table cell. "
)


Questions = [
    "List the rainfall values for each day in January. ",
    "List the rainfall values for each day in February. ",
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
                {"type": "image", "url": url1},
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
        generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
