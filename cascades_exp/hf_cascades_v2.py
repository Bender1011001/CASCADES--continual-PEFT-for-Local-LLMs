import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader

# ============================================================================
# CASCADES v2: Full-Fusion Edition
# Integrates: CaLoRA PaCA (Intersection B), DEAL Heat Kernel (Intersection B),
#             CoSO Null-Space (Intersection C), plus v1's Riemannian QR + SVC
# ============================================================================

# --- Intersection B: CaLoRA PaCA Causal Attribution ---
def paca_causal_mask(grad, historical_grads=None, temperature=0.1):
    """Lightweight PaCA approximation from CaLoRA NeurIPS'25.
    Uses Taylor-expansion proxy: gradient magnitude weighted by cosine correlation
    to historical task gradients. Identifies which parameters causally contributed
    to past task performance and should be protected."""
    if historical_grads is None or len(historical_grads) == 0:
        return torch.ones_like(grad)
    corr = torch.stack([
        F.cosine_similarity(grad.flatten(), hg.flatten(), dim=0) 
        for hg in historical_grads
    ]).mean()
    # High correlation → protect (mask near 0); low correlation → update freely (mask near 1)
    inv_importance = 1.0 - torch.sigmoid(
        (grad.abs() / (grad.abs().mean() + 1e-8)) * (1 + corr) / temperature
    )
    return inv_importance

# --- Intersection B: DEAL Heat Kernel Low-Pass Filter ---
def deal_heat_kernel_filter(grad, t=0.1, lambda_decay=0.01):
    """DEAL NeurIPS'25 wavelet/heat-kernel low-pass filter on gradients.
    Preserves low-frequency generalizable structure across tasks,
    kills high-frequency task-specific noise that causes forgetting.
    Uses fast SVD approximation on the gradient matrix."""
    if grad.dim() < 2:
        return grad
    u, s, v = torch.svd(grad.float())
    decay = torch.exp(-lambda_decay * t * torch.arange(len(s), device=s.device, dtype=torch.float32) ** 2)
    s_filtered = s * decay
    return (u @ torch.diag(s_filtered) @ v.T).to(grad.dtype)


class CASCADES_v2_Adapter(nn.Module):
    """Full-fusion CASCADES adapter with all intersection mechanisms."""
    
    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        
        # Shared Live Subspace — scaled for 4-bit stability
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        self.task_lambdas = nn.ParameterDict()
        
        # EMA Gradient Tracking (GORP-style)
        self.register_buffer('ema_U', torch.zeros_like(self.U_shared))
        self.register_buffer('ema_V', torch.zeros_like(self.V_shared))
        self.beta1 = 0.9
        
        # Intersection B: Historical gradient storage for PaCA
        self.historical_U_grads = []
        self.historical_V_grads = []
        
        # Intersection C: CoSO Frequent Directions sketch for null-space
        self.register_buffer('null_sketch_U', torch.zeros(out_features, rank))
        self.register_buffer('null_sketch_V', torch.zeros(rank, in_features))
        self.null_space_initialized = False
        self.current_task_id = 0

    def add_task(self, task_id):
        self.task_lambdas[str(task_id)] = nn.Parameter(
            torch.ones(self.r, self.r, device=self.U_shared.device)
        )

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = getattr(self, "current_task_id", 0)
        Lam = self.task_lambdas[str(task_id)]
        return (x.to(torch.float32) @ self.V_shared.T @ Lam.T @ self.U_shared.T).to(x.dtype)
    
    def store_task_gradients(self):
        """Called at end of each task to snapshot gradients for PaCA attribution."""
        with torch.no_grad():
            if self.U_shared.grad is not None:
                self.historical_U_grads.append(self.ema_U.clone().flatten())
                if len(self.historical_U_grads) > 5:
                    self.historical_U_grads.pop(0)
            if self.V_shared.grad is not None:
                self.historical_V_grads.append(self.ema_V.clone().flatten())
                if len(self.historical_V_grads) > 5:
                    self.historical_V_grads.pop(0)
    
    def update_null_space_sketch(self):
        """CoSO-style Frequent Directions null-space sketch update.
        Accumulates the subspace of past task directions to project future
        gradients into the orthogonal complement (null-space protection)."""
        with torch.no_grad():
            if self.U_shared.grad is not None:
                # Rank-1 sketch update via outer product of EMA gradient direction
                direction = self.ema_U / (self.ema_U.norm() + 1e-8)
                alpha = 0.5 if self.null_space_initialized else 1.0
                self.null_sketch_U.mul_(alpha).add_(direction, alpha=1.0 - alpha)
                self.null_space_initialized = True
    
    def apply_null_space_projection(self):
        """Project current gradients into the null-space of historical task directions.
        This prevents updates that would interfere with previously learned tasks."""
        with torch.no_grad():
            if self.U_shared.grad is not None and self.null_space_initialized:
                # QR-based orthonormal basis of the historical sketch
                Q, _ = torch.linalg.qr(self.null_sketch_U)
                # Project gradient into null-space: g' = g - Q @ Q^T @ g
                proj = Q @ (Q.T @ self.U_shared.grad)
                self.U_shared.grad.sub_(proj)

    def hamiltonian_descent_step(self, lr=0.01):
        """Full v2 Hamiltonian descent with PaCA + DEAL + QR retraction."""
        with torch.no_grad():
            if self.U_shared.grad is not None and self.V_shared.grad is not None:
                # --- Intersection B: PaCA Causal Mask ---
                mask_U = paca_causal_mask(
                    self.U_shared.grad.flatten(), self.historical_U_grads
                ).reshape(self.U_shared.shape)
                mask_V = paca_causal_mask(
                    self.V_shared.grad.flatten(), self.historical_V_grads
                ).reshape(self.V_shared.shape)
                
                masked_grad_U = self.U_shared.grad * mask_U
                masked_grad_V = self.V_shared.grad * mask_V
                
                # --- Intersection B: DEAL Heat Kernel Filter ---
                filtered_U = deal_heat_kernel_filter(masked_grad_U)
                filtered_V = deal_heat_kernel_filter(masked_grad_V)
                
                # EMA tracking on filtered gradients
                self.ema_U.mul_(self.beta1).add_(filtered_U, alpha=1 - self.beta1)
                self.ema_V.mul_(self.beta1).add_(filtered_V, alpha=1 - self.beta1)
                
                # First-order Riemannian projection + QR retraction (v1 proven)
                grad_U_proj = self.ema_U - self.U_shared @ (self.ema_U.T @ self.U_shared)
                self.U_shared.sub_(lr * grad_U_proj)
                Q_U, _ = torch.linalg.qr(self.U_shared)
                self.U_shared.copy_(Q_U)
                
                grad_V_proj = self.ema_V - (self.ema_V @ self.V_shared.T) @ self.V_shared
                self.V_shared.sub_(lr * grad_V_proj)
                Q_V, _ = torch.linalg.qr(self.V_shared.T)
                self.V_shared.copy_(Q_V.T)
                
                # SVC Calibration
                active_tid = str(self.current_task_id)
                if active_tid in self.task_lambdas:
                    Lam = self.task_lambdas[active_tid]
                    U_s, S, V_s = torch.svd(Lam)
                    S = S / (1 + self.svc_lambda * S)
                    Lam.copy_(U_s @ torch.diag(S) @ V_s.T)


class CASCADES_v2_Linear(nn.Module):
    """Wraps a linear layer with the full CASCADES v2 adapter."""
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = CASCADES_v2_Adapter(self.in_features, self.out_features, rank=rank)
        self.current_task_id = 0
        
    def add_task(self, task_id):
        self.adapter.add_task(task_id)

    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs) if not isinstance(self.base_layer, nn.Linear) else self.base_layer(x)
        self.adapter.current_task_id = self.current_task_id
        adapt_out = self.adapter(x)
        return base_out + 0.1 * adapt_out


def inject_cascades_v2(model, rank=8, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]):
    adapters = []
    for name, module in dict(model.named_modules()).items():
        if any(t in name for t in target_modules) and (isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"):
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            try:
                parent = model.get_submodule(parent_name)
                new_module = CASCADES_v2_Linear(module, rank=rank)
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
    """Distinct continual learning tasks using differentiated prompt distributions."""
    torch.manual_seed(42 + task_number)
    
    task_prompts = [
        # Task 0: Product reviews (sentiment)
        [
            "Review: This product exceeded my expectations. Rating: Positive",
            "Review: Terrible quality, broke after one day. Rating: Negative",
            "Review: Amazing value for the price. Rating: Positive",
            "Review: Would not recommend to anyone. Rating: Negative",
            "Review: Best purchase I've made this year. Rating: Positive",
            "Review: Complete waste of money. Rating: Negative",
        ],
        # Task 1: Movie reviews (different domain, similar structure)
        [
            "Film critique: The cinematography was breathtaking. Verdict: Positive",
            "Film critique: Worst screenplay I've ever seen. Verdict: Negative",
            "Film critique: Outstanding performances by the entire cast. Verdict: Positive",
            "Film critique: A boring and predictable storyline. Verdict: Negative",
            "Film critique: A masterpiece of modern cinema. Verdict: Positive",
            "Film critique: Completely unwatchable garbage. Verdict: Negative",
        ],
        # Task 2: Restaurant reviews (third domain)
        [
            "Dining experience: The flavors were absolutely divine. Score: Positive",
            "Dining experience: Food was cold and service was rude. Score: Negative",
            "Dining experience: Best sushi I've ever had. Score: Positive",
            "Dining experience: Overpriced and underwhelming. Score: Negative",
            "Dining experience: A perfect evening of fine dining. Score: Positive",
            "Dining experience: Found a hair in my soup. Score: Negative",
        ],
    ]
    
    prompts = task_prompts[task_number % len(task_prompts)] * 8  # 48 samples per task
    encodings = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=64)
    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    return DataLoader(dataset, batch_size=2, shuffle=True)


def train_cascades_v2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CASCADES v2 Full-Fusion on {device} ===")
    
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
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    
    adapters = inject_cascades_v2(model, rank=8)
    print(f"Injected CASCADES v2 into {len(adapters)} layers.")
    
    num_tasks = 3
    epochs = 3
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()
    
    for t in range(num_tasks):
        print(f"\n{'='*50}")
        print(f"--- Training CASCADES v2 Task {t} ---")
        print(f"{'='*50}")
        
        for adapter in adapters:
            adapter.add_task(t)
        
        dataloader = prepare_data(tokenizer, t)
        
        optimizer = optim.Adam([
            {'params': [a.U_shared for a in adapters] + [a.V_shared for a in adapters], 'lr': 1e-4},
            {'params': [a.task_lambdas[str(t)] for a in adapters], 'lr': 5e-3}
        ])
        
        for name, module in model.named_modules():
            if isinstance(module, CASCADES_v2_Linear):
                module.current_task_id = t
        
        for ep in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                adapter_params = [p for a in adapters for p in a.parameters() if p.grad is not None]
                if adapter_params:
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                
                # === CASCADES v2 Full Pipeline ===
                for a in adapters:
                    # Intersection C: CoSO null-space projection (protect past tasks)
                    if t > 0:
                        a.apply_null_space_projection()
                    
                    # Intersection B + v1 core: PaCA + DEAL + Riemannian QR + SVC
                    a.hamiltonian_descent_step()
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        
        # End of task: store gradients for PaCA + update null-space sketch
        for a in adapters:
            a.store_task_gradients()
            a.update_null_space_sketch()
        
        # Evaluation
        print(f"\n--- Evaluation after Task {t} ---")
        for eval_t in range(t + 1):
            eval_dataloader = prepare_data(tokenizer, eval_t)
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, CASCADES_v2_Linear):
                        module.current_task_id = eval_t
                
                for batch in eval_dataloader:
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(device)
                    out = model(input_ids=input_ids, labels=input_ids)
                    total_loss += out.loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            proxy_acc = math.exp(-avg_loss)
            accuracy_matrix[t, eval_t] = proxy_acc
            print(f"  Task {eval_t} proxy accuracy: {proxy_acc*100:.2f}% (avg_loss: {avg_loss:.4f})")

    end_time = time.time()
    
    # Final metrics
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)
    
    print(f"\n{'='*50}")
    print("=== FINAL CASCADES v2 METRICS ===")
    print(f"{'='*50}")
    print(f"Average Accuracy Proxy: {avg_acc*100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt*100:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"Accuracy Matrix:")
    for i in range(num_tasks):
        row = " | ".join([f"{accuracy_matrix[i,j]*100:6.2f}%" if accuracy_matrix[i,j] > 0 else "   —   " for j in range(num_tasks)])
        print(f"  After T{i}: {row}")
    
    df = pd.DataFrame(accuracy_matrix,
                      columns=[f"Eval_T{i}" for i in range(num_tasks)],
                      index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("cascades_v2_results.csv")
    print("\nResults saved to cascades_v2_results.csv")

if __name__ == "__main__":
    train_cascades_v2()
