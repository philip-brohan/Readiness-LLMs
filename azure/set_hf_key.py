#!/usr/bin/env python

# Store Philip's HF Key in the Azure key-vault

import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

keyVaultName = "kv-ai4climate-scratch"
KVUri = f"https://kv-ai4climate-scratch.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

secretName = "Philip-HF-Key"

with open("%s/.huggingface_api" % os.getenv("HOME"), "r") as file:
    secretValue = file.read().strip()

print(
    f"Creating a secret in KV_NAME called '{secretName}' with the value '{secretValue}' ..."
)

client.set_secret(secretName, secretValue, content_type="text/plain")

print(" done.")

print("Retrieving your secret from %s." % keyVaultName)

retrieved_secret = client.get_secret(secretName)

print(f"Your secret is '{retrieved_secret.value}'.")

# print("Deleting your secret from %s ..." % keyVaultName)
# poller = client.begin_delete_secret(secretName)
# deleted_secret = poller.result()

print(" done.")
