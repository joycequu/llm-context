# from huggingface_hub import login
# # login(token='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')
# login(token='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')

import os
from transformers import AutoTokenizer, AutoModel

# Ensure your Hugging Face token is set as an environment variable
token = os.getenv('HUGGINGFACE_TOKEN')
if not token:
    raise ValueError("Hugging Face token not found. Set the HUGGINGFACE_TOKEN environment variable.")

# Define the model name
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "lmsys/vicuna-7b-v1.5"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModel.from_pretrained(model_name, use_auth_token=token)

print("Model and tokenizer successfully downloaded.")