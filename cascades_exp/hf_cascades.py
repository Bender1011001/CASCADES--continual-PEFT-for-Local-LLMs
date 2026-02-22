import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- CASCADES Synthetic Empirical Framework for HuggingFace ---

class CASCADES_Adapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        
        # Shared Live Subspace — scaled down for 4-bit stability
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        self.task_lambdas = nn.ParameterDict()
        
        # EMA Gradient Tracking
        self.register_buffer('ema_U', torch.zeros_like(self.U_shared))
        self.register_buffer('ema_V', torch.zeros_like(self.V_shared))
        self.beta1 = 0.9

    def add_task(self, task_id):
        self.task_lambdas[str(task_id)] = nn.Parameter(torch.ones(self.r, self.r, device=self.U_shared.device))

    def forward(self, x, task_id=None):
        if task_id is None:
            # We assume CASCADES_Linear sets current_task_id before calling
            task_id = getattr(self, "current_task_id", 0)
        Lam = self.task_lambdas[str(task_id)]
        return (x.to(torch.float32) @ self.V_shared.T @ Lam.T @ self.U_shared.T).to(x.dtype)
        
    def hamiltonian_descent_step(self, lr=0.01):
        # Applied as a post-optimizer retraction (Option A)
        with torch.no_grad():
            if self.U_shared.grad is not None and self.V_shared.grad is not None:
                self.ema_U.mul_(self.beta1).add_(self.U_shared.grad, alpha=1 - self.beta1)
                self.ema_V.mul_(self.beta1).add_(self.V_shared.grad, alpha=1 - self.beta1)
                
                # Efficient first-order Riemannian gradient step on Stiefel manifold
                # For dxr matrix U: grad_proj = ema_U - U @ (ema_U^T @ U)
                grad_U_proj = self.ema_U - self.U_shared @ (self.ema_U.T @ self.U_shared)
                self.U_shared.sub_(lr * grad_U_proj)
                Q_U, _ = torch.linalg.qr(self.U_shared)
                self.U_shared.copy_(Q_U)
                
                # For rxd matrix V: grad_proj = ema_V - (ema_V @ V^T) @ V
                grad_V_proj = self.ema_V - (self.ema_V @ self.V_shared.T) @ self.V_shared
                self.V_shared.sub_(lr * grad_V_proj)
                Q_V, _ = torch.linalg.qr(self.V_shared.T)
                self.V_shared.copy_(Q_V.T)
                
                # SVC Calibration: Only run on the currently active task lambda
                active_tid = str(getattr(self, "current_task_id", list(self.task_lambdas.keys())[-1]))
                Lam = self.task_lambdas[active_tid]
                U, S, V = torch.svd(Lam)
                S = S / (1 + self.svc_lambda * S)
                Lam.copy_(U @ torch.diag(S) @ V.T)

class CASCADES_Linear(nn.Module):
    """Wraps an existing nn.Linear (or 4-bit linear) with a CASCADES Adapter."""
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = CASCADES_Adapter(self.in_features, self.out_features, rank=rank)
        self.current_task_id = 0
        
    def add_task(self, task_id):
        self.adapter.add_task(task_id)

    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs) if not isinstance(self.base_layer, nn.Linear) else self.base_layer(x)
        self.adapter.current_task_id = self.current_task_id
        adapt_out = self.adapter(x)
        return base_out + 0.1 * adapt_out  # alpha=0.1 mixing to prevent residual corruption

def inject_cascades(model, rank=8, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]):
    adapters = []
    
    for name, module in dict(model.named_modules()).items():
        if any(target in name for target in target_modules) and (isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            try:
                parent = model.get_submodule(parent_name)
                new_module = CASCADES_Linear(module, rank=rank)
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

def prepare_data(tokenizer, task_number):
    """Mock continual learning data mapping distinct vocab regions for quick 4-bit local testing"""
    torch.manual_seed(42 + task_number)
    prompts = [
        f"Task {task_number}: Evaluate the sentiment. This product is great! -> Positive",
        f"Task {task_number}: Evaluate the sentiment. I hated the delivery. -> Negative",
        f"Task {task_number}: Evaluate the sentiment. Excellent quality. -> Positive",
        f"Task {task_number}: Evaluate the sentiment. Completely broken. -> Negative"
    ] * 10
    
    encodings = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
    
    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    return DataLoader(dataset, batch_size=2, shuffle=True)

def train_hf_cascades():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading 4-bit Qwen3 Heretic model on {device}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model_id = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    adapters = inject_cascades(model, rank=8)
    print(f"Injected CASCADES into {len(adapters)} linear layers.")
    
    num_tasks = 3 
    epochs = 3
    
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()
    
    for t in range(num_tasks):
        print(f"\n--- Training CASCADES Sequence Task {t} ---")
        
        for adapter in adapters:
            adapter.add_task(t)
            
        dataloader = prepare_data(tokenizer, t)
        
        optimizer = optim.Adam([
            {'params': [a.U_shared for a in adapters] + [a.V_shared for a in adapters], 'lr': 1e-4}, 
            {'params': [a.task_lambdas[str(t)] for a in adapters], 'lr': 5e-3}
        ])
        
        for name, module in model.named_modules():
            if isinstance(module, CASCADES_Linear):
                module.current_task_id = t
        
        for ep in range(epochs):
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping for 4-bit stability
                adapter_params = [p for a in adapters for p in a.parameters() if p.grad is not None]
                if adapter_params:
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                
                # CASCADES Mathematical Updates
                for a in adapters:
                    a.hamiltonian_descent_step()
                    
                optimizer.step()
                
            print(f"Task {t}, Epoch {ep+1}/{epochs}, Loss: {loss.item():.4f}")
            
        # Simplified Eval tracking (reconstruction loss as proxy for accuracy in generative)
        for eval_t in range(t + 1):
             eval_dataloader = prepare_data(tokenizer, eval_t)
             total_loss = 0
             num_batches = 0
             with torch.no_grad():
                 # Set task_id on CASCADES_Linear modules (not just adapters)
                 for name, module in model.named_modules():
                     if isinstance(module, CASCADES_Linear):
                         module.current_task_id = eval_t
                 
                 for batch in eval_dataloader:
                     input_ids, attention_mask = batch
                     input_ids = input_ids.to(device)
                     
                     out = model(input_ids=input_ids, labels=input_ids)
                     total_loss += out.loss.item()
                     num_batches += 1
             
             # exp(-avg_CE_loss) maps CE loss to (0,1] proxy accuracy
             avg_loss = total_loss / max(num_batches, 1)
             proxy_acc = math.exp(-avg_loss)
             accuracy_matrix[t, eval_t] = proxy_acc
             print(f"Performance proxy on Task {eval_t}: {proxy_acc*100:.2f}%")

    end_time = time.time()
    
    # Compute Metrics
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)
    
    print("\n--- FINAL HF CASCADES METRICS ---")
    print(f"Average Accuracy Proxy: {avg_acc*100:.2f}%")
    print(f"Backward Transfer (BWT proxy): {bwt*100:.2f}%")
    print(f"Total Computation Time: {end_time - start_time:.2f}s")
    
    df = pd.DataFrame(accuracy_matrix, columns=[f"Eval_T{i}" for i in range(num_tasks)], index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("hf_cascades_exp_results.csv")
    print("Results saved to hf_cascades_exp_results.csv")

if __name__ == "__main__":
    train_hf_cascades()
