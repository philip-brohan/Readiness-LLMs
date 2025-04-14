# Utility functions for interacting with huggingface

import os
from huggingface_hub import login


# Get the Huggingface API key - either file or environment variable
def HFlogin():
    """Login to Huggingface using the API key."""
    if "HF_KEY" in os.environ:
        hf_key = os.getenv("HF_KEY")
    else:
        # If not found in environment variable, look for file
        try:
            with open("%s/.huggingface_api" % os.getenv("HOME"), "r") as file:
                hf_key = file.read().strip()
        except FileNotFoundError:
            raise Exception("Huggingface API key not found.")
    login(token=hf_key)
