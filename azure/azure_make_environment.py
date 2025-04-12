#!/usr/bin/env python

# This script makes an azure compute environment for running Llama on the A100 node

import os
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment


# Connect using Default Credential
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    token = credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    raise Exception("Login with `az login' before running this script")

# Authenticate
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ.get("AZML_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZML_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZML_WORKSPACE_NAME"),
)

# Define the environment from a conda yml file
bindir = os.path.abspath(os.path.dirname(__file__))
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="%s/../environment/llm-azure.yml" % bindir,
    name="LLM-Azure",
    description="LLM environment",
)

# Update the environment
ml_client.environments.create_or_update(env_docker_conda)

print("\nAvailable environments (examples):")
for e in ml_client.environments.list():
   if (
       e.creation_context.created_by_type == "User"
       and e.creation_context.created_by != "Microsoft"
   ):
       print(e.name, e.latest_version)

# Note. You can't delete an environment - archive them instead.

# e.g. ml_client.environments.archive('MLP-Azure')
