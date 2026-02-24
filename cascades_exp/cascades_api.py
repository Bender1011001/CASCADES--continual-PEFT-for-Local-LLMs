import torch
import math
import numpy as np
import os
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from hf_cascades_reasoning import CASCADES_v6_Linear, CASCADES_v6_Adapter

app = Flask(__name__)
CORS(app) # Allow local web UI to connect

# ==========================================
# 1. Global State & Initialization
# ==========================================
MODEL_PATH = "Qwen/Qwen2.5-1.5B" 
WEIGHTS_PATH = "cascades_v9_weights.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Starting CASCADES Inference API...")

# Configure 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"Loading Base Model ({MODEL_PATH}) in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Disable caching for custom forward pass compatibility
model.config.use_cache = False

# ==========================================
# 2. Inject Custom CASCADES Architecture
# ==========================================
print("Injecting CASCADES v9 Manifold Architecture...")
target_modules = ["q_proj", "v_proj", "up_proj"]

for name, module in dict(model.named_modules()).items():
    if not hasattr(module, "weight"):
        continue

    is_target = any(t in name for t in target_modules)
    if is_target:
        # We assume standard 8GB config from training
        wrapper = CASCADES_v6_Linear(module, rank=8, svc_lambda=0.01)
        
        # We have to replace the module in the parent
        parent_name = name.rsplit('.', 1)[0]
        child_name = name.rsplit('.', 1)[-1]
        
        if parent_name == '':
            setattr(model, child_name, wrapper)
        else:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, wrapper)

print(f"Loading trained weights from {WEIGHTS_PATH}...")
try:
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    # Filter state dict to only include loaded modules
    current_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in current_state}
    model.load_state_dict(filtered_state, strict=False)
    print("Weights loaded successfully!")
except FileNotFoundError:
    print(f"WARNING: '{WEIGHTS_PATH}' not found. Serving untrained (randomized) CASCADES layers.")

model.eval()

# ==========================================
# 3. API Endpoints
# ==========================================

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Generates a response using the CASCADES enhanced model."""
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_tokens', 256)
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    print(f"Received Prompt: {prompt[:50]}...")
    
    # Format for Qwen chat
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    # Disable gradients for pure inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    print("\nAPI is running on http://127.0.0.1:5000")
    print("WARNING: First inference pass may be slow due to CUDA graph compilation.")
    app.run(host='0.0.0.0', port=5000)
