import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from hf_cascades_reasoning import CASCADES_v6_Linear

def evaluate_exact_match():
    MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
    WEIGHTS_PATH = "e:/code.projects/research/cascades_v9_weights.pt"
    DATA_PATH = "e:/code.projects/research/cascades_exp/task2_action_cot.jsonl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Configuring 4-bit precision...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print("Injecting CASCADES v9 Manifold Architecture...")
    target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

    for name, module in dict(model.named_modules()).items():
        if not hasattr(module, "weight"): continue
        if any(t in name for t in target_modules) and (isinstance(module, torch.nn.Linear) or type(module).__name__ == "Linear4bit"):
            # Assume strict critical injection for inference
            wrapper = CASCADES_v6_Linear(module, rank=8, is_critical=True)
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            if parent_name == '':
                setattr(model, child_name, wrapper)
            else:
                setattr(model.get_submodule(parent_name), child_name, wrapper)

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading trained weights from {WEIGHTS_PATH}...")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        current_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in current_state}
        model.load_state_dict(filtered_state, strict=False)
        print("Weights loaded successfully!")
    else:
        print(f"WARNING: '{WEIGHTS_PATH}' not found. Using untrained weights.")

    model.eval()

    print("Loading 50 hold-out samples...")
    samples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            samples.append(json.loads(line))
            if len(samples) >= 50:
                break
                
    # If file had less than 50 samples
    samples = samples[:50]
    
    exact_matches = 0
    total = len(samples)
    print(f"Starting generative evaluation on {total} samples...")

    for i, sample in enumerate(samples):
        prompt = sample['prompt']
        target_response = sample['response'].strip()
        
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        if generated_text == target_response:
            exact_matches += 1
            print(f"[{i+1}/{total}] MATCH")
        else:
            print(f"[{i+1}/{total}] MISMATCH")
            
    em_rate = (exact_matches / total) * 100
    print(f"\nFinal Exact Match Rate: {em_rate:.2f}% ({exact_matches}/{total})")

if __name__ == "__main__":
    evaluate_exact_match()
