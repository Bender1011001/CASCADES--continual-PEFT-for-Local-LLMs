import time
import os
import sys

print("Initializing Torch...")
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")

print("Importing HF...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "p-e-w/Qwen3-4B-Instruct-2507-heretic"

print(f"Loading Tokenizer {model_id}...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded in {time.time() - start:.2f}s")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print(f"Loading Model {model_id} in 4-bit...")
start = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    print(f"Model loaded successfully in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Exception during loading: {e}", file=sys.stderr)
