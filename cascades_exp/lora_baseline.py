import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader

# Removed unified dataloader import to use exact CoT dataloader below.

# --- Standard LoRA Baseline for Continual Learning Comparison ---
# Same model, same data, same eval — no Hamiltonian descent, no SVC, no Riemannian tricks

class LoRA_Adapter(nn.Module):
    """Standard LoRA low-rank adapter without any continual learning mechanisms."""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.r = rank
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
    def forward(self, x):
        return (x.to(torch.float32) @ self.B.T @ self.A.T).to(x.dtype)

class LoRA_Linear(nn.Module):
    """Wraps an existing linear layer with a standard LoRA adapter."""
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = LoRA_Adapter(self.in_features, self.out_features, rank=rank)
        
    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs) if not isinstance(self.base_layer, nn.Linear) else self.base_layer(x)
        adapt_out = self.adapter(x)
        return base_out + 0.1 * adapt_out  # Same alpha=0.1 mixing as CASCADES for fairness

def inject_lora(model, rank=8, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]):
    adapters = []
    
    for name, module in dict(model.named_modules()).items():
        if any(target in name for target in target_modules) and (isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            try:
                parent = model.get_submodule(parent_name)
                new_module = LoRA_Linear(module, rank=rank)
                new_module = new_module.to(module.weight.device)
                setattr(parent, child_name, new_module)
                adapters.append(new_module.adapter)
            except AttributeError:
                pass
                
    for param in model.parameters():
        param.requires_grad = False
        
    for adapter in adapters:
        for param in adapter.parameters():
            param.requires_grad = True
            
    return adapters

def prepare_data(tokenizer, task_number, base_seed=42):
    """Loads domain-specific JSONL prompts mapped for Llama-3-8B CoT baseline adaptation."""
    import os
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(base_seed + task_number)

    # Route to the distilled explicit CoT files
    files = [
        "cascades_exp/task0_logic_cot.jsonl",
        "cascades_exp/task1_decomp_cot.jsonl",
        "cascades_exp/task2_action_cot.jsonl"
    ]
    file_path = files[task_number % len(files)]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset: {file_path}. Please run format_cot_datasets.py first.")

    df = pd.read_json(file_path, lines=True)
    input_ids_list, attention_masks_list, labels_list = [], [], []
    
    # Context window raised to safely envelop lengthier step-by-step traces
    max_length = 1024 
    
    for p, r in zip(df['prompt'], df['response']):
        # Llama-3 Native Chat Templating format 
        messages = [{"role": "user", "content": p}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response_text = r + tokenizer.eos_token
        
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_tokens = tokenizer(response_text, add_special_tokens=False).input_ids
        
        input_ids = prompt_tokens + response_tokens
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            
        attention_mask = [1] * len(input_ids)
        
        # Strict Autoregressive Masking globally isolates reasoning paths
        labels = [-100] * len(prompt_tokens) + response_tokens
        if len(labels) > max_length:
            labels = labels[:max_length]
            
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length
            
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(labels)
        
    dataset = TensorDataset(
        torch.tensor(input_ids_list), 
        torch.tensor(attention_masks_list), 
        torch.tensor(labels_list)
    )
    return DataLoader(dataset, batch_size=1, shuffle=True)

def train_lora_baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading 4-bit Llama-3-8B model on {device} (LoRA baseline)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    adapters = inject_lora(model, rank=8)
    print(f"Injected LoRA into {len(adapters)} linear layers.")
    
    num_tasks = 3
    epochs = 3
    
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()
    
    for t in range(num_tasks):
        print(f"\n--- Training LoRA Baseline Task {t} ---")
        
        dataloader = prepare_data(tokenizer, t)
        
        # Standard Adam on ALL adapter parameters — no task isolation
        optimizer = optim.Adam([p for a in adapters for p in a.parameters()], lr=1e-4)
        
        for ep in range(epochs):
            for i, batch in enumerate(dataloader):
                print(f"    [Ep {ep+1} B {i+1}] Starting batch...")
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                print(f"    [Ep {ep+1} B {i+1}] Forward pass...")
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                print(f"    [Ep {ep+1} B {i+1}] Backward pass...")
                loss.backward()
                
                # Same grad clipping as CASCADES for fairness
                adapter_params = [p for a in adapters for p in a.parameters() if p.grad is not None]
                if adapter_params:
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                
                # Standard Adam step — NO Hamiltonian descent, NO SVC calibration
                optimizer.step()
                print(f"    [Ep {ep+1} B {i+1}] Batch finished.")
                
            print(f"Task {t}, Epoch {ep+1}/{epochs}, Avg Loss: {loss.item():.4f}")
            
        # Eval (same method as CASCADES)
        for eval_t in range(t + 1):
             eval_dataloader = prepare_data(tokenizer, eval_t)
             total_loss = 0
             num_batches = 0
             with torch.no_grad():
                 for batch in eval_dataloader:
                     input_ids, attention_mask, labels = batch
                     input_ids, labels = input_ids.to(device), labels.to(device)
                     
                     out = model(input_ids=input_ids, labels=labels)
                     total_loss += out.loss.item()
                     num_batches += 1
             
             avg_loss = total_loss / max(num_batches, 1)
             proxy_acc = math.exp(-avg_loss)
             accuracy_matrix[t, eval_t] = proxy_acc
             print(f"Performance proxy on Task {eval_t}: {proxy_acc*100:.2f}%")

    end_time = time.time()
    
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)
    
    print("\n--- FINAL LORA BASELINE METRICS ---")
    print(f"Average Accuracy Proxy: {avg_acc*100:.2f}%")
    print(f"Backward Transfer (BWT proxy): {bwt*100:.2f}%")
    print(f"Total Computation Time: {end_time - start_time:.2f}s")
    
    df = pd.DataFrame(accuracy_matrix, columns=[f"Eval_T{i}" for i in range(num_tasks)], index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("lora_baseline_results.csv")
    print("Results saved to lora_baseline_results.csv")

if __name__ == "__main__":
    train_lora_baseline()
