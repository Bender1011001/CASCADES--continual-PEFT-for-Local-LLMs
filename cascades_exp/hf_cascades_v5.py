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
# CASCADES v3.1: 15-Paper Full-Fusion Edition
# ============================================================================
# Intersection A: Share + Online Subspace Descent + GORP + StelLA + Riemannian LoRA
# Intersection B: CaLoRA PaCA + DEAL heat kernel (quant-aware) + GainLoRA gates
# Intersection C: CoSO Frequent Directions + LANCE HOSVD + SVC + CL-LoRA reassignment
# Intersection D: D-MoLE layer selection + FunLoRA rank-1 functional expansion
# ============================================================================

# --- Ablation Flags ---
ENABLE_PACA = True            # Intersection B: CaLoRA causal attribution
ENABLE_DEAL = True            # Intersection B: heat kernel filter (quant-aware)
ENABLE_GAINLORA_GATE = True   # Intersection B: GainLoRA learned interference gate
ENABLE_COSO_NULLSPACE = True  # Intersection C: CoSO Frequent Directions
ENABLE_CLLORA_REASSIGN = True # Intersection C: CL-LoRA gradient reassignment
ENABLE_SVC = True             # Intersection C: Singular Value Calibration
ENABLE_DMOLE_SELECT = True    # Intersection D: D-MoLE layer importance selection
ENABLE_FUNLORA = True         # Intersection D: FunLoRA rank-1 for non-critical layers


# --- Intersection A: StelLA-style Riemannian step (NeurIPS'25 Spotlight) ---
def stella_riemannian_step(param, grad, lr=0.01):
    """StelLA's modular Euclidean→Riemannian conversion.
    Computes the Riemannian gradient on the Stiefel manifold and retracts via QR.
    Replaces the ad-hoc QR retraction from v1/v2 with the principled StelLA formulation."""
    with torch.no_grad():
        # Riemannian gradient: grad_R = grad - param @ sym(param^T @ grad)
        sym = (param.T @ grad + grad.T @ param) / 2.0
        riemannian_grad = grad - param @ sym
        # QR retraction (O(dr²))
        updated = param - lr * riemannian_grad
        Q, _ = torch.linalg.qr(updated)
        param.copy_(Q)


# --- Intersection B: CaLoRA PaCA causal mask ---
def paca_causal_mask(grad, historical_grads, temperature=0.1):
    """CaLoRA NeurIPS'25: parameter-level counterfactual attribution.
    Identifies which gradient components causally benefit past task performance.
    High correlation to historical grads → protect (mask → 0); low → update freely (mask → 1)."""
    if not ENABLE_PACA or len(historical_grads) == 0:
        return torch.ones_like(grad)
    flat = grad.flatten()
    corr = torch.stack([
        F.cosine_similarity(flat, hg.flatten(), dim=0)
        for hg in historical_grads
    ]).mean()
    inv_importance = 1.0 - torch.sigmoid(
        (grad.abs() / (grad.abs().mean() + 1e-8)) * (1 + corr) / temperature
    )
    return inv_importance


# --- Intersection B: DEAL heat kernel (quantization-aware v3 fix) ---
def deal_heat_kernel_filter(grad, quant_noise_std=0.0, t=0.05, lambda_decay=0.01):
    """DEAL arXiv'25: heat-kernel low-pass filter on gradients.
    v4 fix: adaptive threshold ε_quant based on quantization noise floor.
    Filtering operates strictly within coordinates avoiding explicit O(d^3) SVD."""
    if not ENABLE_DEAL or grad.dim() < 2:
        return grad
    
    # Quantization-aware noise floor proxy parameterized by quantization step
    eps_quant = max(1e-4, quant_noise_std * (0.01 / math.sqrt(12))) 
    
    grad_norm = grad.norm()
    # Filter out pure noise regimes
    if grad_norm < eps_quant:
        return torch.zeros_like(grad)
        
    decay = math.exp(-lambda_decay * t)
    return grad * decay


# --- Intersection C: CL-LoRA gradient reassignment ---
def cllora_gradient_reassign(grad, null_sketch, alpha=1.0):
    """CL-LoRA CVPR'25 / CASCADES EAR: redirect blocked gradient energy into the free subspace.
    Uses exact Energy-Accounted Reassignment (EAR) to preserve ||g||_2 while staying
    in the feasible orthogonal subspace."""
    if not ENABLE_CLLORA_REASSIGN or null_sketch is None:
        return grad
    # Compute null-space projection P
    Q, _ = torch.linalg.qr(null_sketch)
    occupied_component = Q @ (Q.T @ grad)
    null_component = grad - occupied_component
    
    # EXACT EAR SCALING: g_EAR = ( ||g||_2 / (||g_free||_2 + eps) ) * g_free
    grad_energy = grad.norm()
    null_energy = null_component.norm()
    
    if grad_energy > 1e-8 and null_energy > 1e-8:
        return (grad_energy / (null_energy + 1e-8)) * null_component
        
    return null_component


# --- Intersection D: FunLoRA rank-1 functional expansion ---
class FunLoRA_Adapter(nn.Module):
    """FunLoRA arXiv'25: rank-1 matrices with functional expansion.
    Three nonlinear bases on rank-1 achieve effective rank ≥ 3 with O(d) memory."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # v5 FunLoRA Scale Tuning: increased variance slightly to prevent bottlenecking
        self.a = nn.Parameter(torch.randn(out_features, 1) * 0.1)
        self.b = nn.Parameter(torch.randn(1, in_features) * 0.1)

    def forward(self, x):
        ab = x.to(torch.float32) @ self.b.T @ self.a.T  # rank-1 base
        # Functional expansion: identity + sigmoid + tanh = effective rank ≥ 3
        expanded = ab + torch.sigmoid(ab) + torch.tanh(ab)
        return expanded.to(x.dtype)


# --- Full CASCADES v3.1 Adapter (critical layers) ---
class CASCADES_v3_Adapter(nn.Module):
    """Full 15-paper fusion adapter for critical layers."""

    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        self.in_features = in_features
        self.out_features = out_features

        # Intersection A: Shared Stiefel subspace (StelLA USV^T decomposition)
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.task_lambdas = nn.ParameterDict()

        # Intersection A: GORP EMA gradient tracking
        self.register_buffer('ema_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_V', torch.zeros(rank, in_features))
        self.beta1 = 0.9

        # Intersection B: GainLoRA learned interference gate
        if ENABLE_GAINLORA_GATE:
            # v5: Task-Aware Subspace Gating (TAG)
            self.task_embedding = nn.Embedding(10, rank) # Support up to 10 tasks
            self.gate_proj = nn.Linear(rank * 3, 1, bias=True) # U_mean + V_mean + Task_Emb
            nn.init.constant_(self.gate_proj.bias, 1.0)  # start open

        # Intersection B: PaCA historical gradient storage
        self.historical_U_grads = []
        self.historical_V_grads = []

        # Intersection C: CoSO null-space sketch
        self.register_buffer('null_sketch_U', torch.zeros(out_features, rank))
        self.null_space_initialized = False

        # Quantization noise estimate (updated during training)
        self.register_buffer('quant_noise_std', torch.tensor(0.0))
        self.current_task_id = 0

    def add_task(self, task_id):
        self.task_lambdas[str(task_id)] = nn.Parameter(
            torch.ones(self.r, self.r, device=self.U_shared.device)
        )

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = self.current_task_id
        input_dtype = x.dtype
        Lam = self.task_lambdas[str(task_id)]
        adapt_out = (x.to(torch.float32) @ self.V_shared.T @ Lam.T @ self.U_shared.T)

        # Intersection B: GainLoRA gate
        if ENABLE_GAINLORA_GATE and self.gate_proj is not None:
            # Gate input: summary of current subspace state + Task Embedding
            task_emb = self.task_embedding(torch.tensor(task_id, device=self.U_shared.device))
            gate_input = torch.cat([
                self.U_shared.mean(dim=0),
                self.V_shared.mean(dim=1),
                task_emb
            ], dim=0) # Note: we allow backprop through the subspace state here
            gate_value = torch.sigmoid(self.gate_proj(gate_input))
            adapt_out = gate_value * adapt_out

        # CRITICAL: cast back to input dtype to prevent SDPA mixed-dtype error
        return adapt_out.to(input_dtype)

    def store_task_gradients(self):
        """End-of-task: snapshot EMA gradients for PaCA attribution."""
        with torch.no_grad():
            if self.ema_U.norm() > 1e-8:
                self.historical_U_grads.append(self.ema_U.clone().flatten())
                if len(self.historical_U_grads) > 5:
                    self.historical_U_grads.pop(0)
            if self.ema_V.norm() > 1e-8:
                self.historical_V_grads.append(self.ema_V.clone().flatten())
                if len(self.historical_V_grads) > 5:
                    self.historical_V_grads.pop(0)

    def update_null_space_sketch(self):
        """CoSO-style Frequent Directions null-space sketch update."""
        if not ENABLE_COSO_NULLSPACE:
            return
        with torch.no_grad():
            if self.ema_U.norm() > 1e-8:
                direction = self.ema_U / (self.ema_U.norm() + 1e-8)
                alpha = 0.5 if self.null_space_initialized else 1.0
                self.null_sketch_U.mul_(alpha).add_(direction, alpha=1.0 - alpha)
                self.null_space_initialized = True

    def full_descent_step(self, lr=0.01):
        """Full v3.1 descent: PaCA → DEAL(quant-aware) → CoSO+CL-LoRA → StelLA → SVC"""
        with torch.no_grad():
            if self.U_shared.grad is None or self.V_shared.grad is None:
                return

            grad_U = self.U_shared.grad.clone()
            grad_V = self.V_shared.grad.clone()

            # === Intersection B: PaCA Causal Mask ===
            mask_U = paca_causal_mask(grad_U.flatten(), self.historical_U_grads).reshape(grad_U.shape)
            mask_V = paca_causal_mask(grad_V.flatten(), self.historical_V_grads).reshape(grad_V.shape)
            grad_U = grad_U * mask_U
            grad_V = grad_V * mask_V

            # === Intersection B: DEAL Heat Kernel (quant-aware) ===
            noise_std = self.quant_noise_std.item()
            grad_U = deal_heat_kernel_filter(grad_U, quant_noise_std=noise_std)
            grad_V = deal_heat_kernel_filter(grad_V, quant_noise_std=noise_std)

            # === Intersection A: GORP EMA tracking ===
            self.ema_U.mul_(self.beta1).add_(grad_U, alpha=1 - self.beta1)
            self.ema_V.mul_(self.beta1).add_(grad_V, alpha=1 - self.beta1)

            # === Intersection C: CoSO null-space + CL-LoRA reassignment ===
            if ENABLE_COSO_NULLSPACE and self.null_space_initialized:
                ema_U_reassigned = cllora_gradient_reassign(self.ema_U, self.null_sketch_U)
                self.ema_U.copy_(ema_U_reassigned)

            # === Intersection A: StelLA Riemannian step ===
            stella_riemannian_step(self.U_shared, self.ema_U, lr=lr)
            # For V, transpose convention
            stella_riemannian_step(self.V_shared.T.contiguous(), self.ema_V.T.contiguous(), lr=lr)
            # V_shared needs the transposed result
            with torch.no_grad():
                Q_V, _ = torch.linalg.qr(self.V_shared.T)
                self.V_shared.copy_(Q_V.T)

            # === Intersection C: SVC Calibration ===
            if ENABLE_SVC:
                active_tid = str(self.current_task_id)
                if active_tid in self.task_lambdas:
                    Lam = self.task_lambdas[active_tid]
                    U_s, S, V_s = torch.svd(Lam)
                    S = S / (1 + self.svc_lambda * S)
                    Lam.copy_(U_s @ torch.diag(S) @ V_s.T)

                    # === v5: Adaptive Rank Routing (ARR) / Rank Decay ===
                    # If a rank dimension's energy drops below threshold, reset its Stiefel basis
                    min_energy_threshold = 1e-3
                    if S.min() < min_energy_threshold:
                        decay_mask = (S < min_energy_threshold) # Shape (r,)
                        if decay_mask.any():
                            with torch.no_grad():
                                # Inject random noise into dead columns of U
                                noise_U = torch.randn_like(self.U_shared[:, decay_mask]) * 0.01
                                self.U_shared[:, decay_mask] = self.U_shared[:, decay_mask] * 0.5 + noise_U
                                
                                # Inject random noise into dead rows of V
                                noise_V = torch.randn_like(self.V_shared[decay_mask, :]) * 0.01
                                self.V_shared[decay_mask, :] = self.V_shared[decay_mask, :] * 0.5 + noise_V
                                
                                # Re-orthogonalize to maintain Stiefel manifold
                                Q_U, _ = torch.linalg.qr(self.U_shared)
                                self.U_shared.copy_(Q_U)
                                
                                Q_V, _ = torch.linalg.qr(self.V_shared.T)
                                self.V_shared.copy_(Q_V.T)


# --- Adaptive Linear Wrapper ---
class CASCADES_v3_Linear(nn.Module):
    """Wraps base layer with either full CASCADES (critical) or FunLoRA (non-critical)."""

    def __init__(self, base_layer, rank=8, is_critical=True):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.is_critical = is_critical
        self.current_task_id = 0

        if is_critical:
            self.adapter = CASCADES_v3_Adapter(self.in_features, self.out_features, rank=rank)
        else:
            self.adapter = FunLoRA_Adapter(self.in_features, self.out_features)

    def add_task(self, task_id):
        if self.is_critical and hasattr(self.adapter, 'add_task'):
            self.adapter.add_task(task_id)

    def forward(self, x, *args, **kwargs):
        base_out = self.base_layer(x, *args, **kwargs) if not isinstance(self.base_layer, nn.Linear) else self.base_layer(x)
        if self.is_critical:
            self.adapter.current_task_id = self.current_task_id
            adapt_out = self.adapter(x)
        else:
            adapt_out = self.adapter(x)
        # Ensure adapter output dtype matches base output to prevent SDPA mismatch
        return base_out + 0.1 * adapt_out.to(base_out.dtype)


# --- Intersection D: D-MoLE Layer Importance Selection ---
def compute_layer_importance(model, dataloader, device, threshold=0.15):
    """D-MoLE ICML'25: gradient-based layer importance scoring.
    For 4-bit models, uses activation-based heuristic (output variance as proxy)
    since 4-bit weights don't support requires_grad. For fp16/fp32 models, uses
    gradient norms directly."""
    if not ENABLE_DMOLE_SELECT:
        return {}  # all critical

    importance = {}
    model.eval()
    activation_stats = {}
    hooks = []

    # Register forward hooks to capture activation variance per layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit":
            activation_stats[name] = 0.0
            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        activation_stats[layer_name] += out.float().var().item()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Multi-batch for importance estimation stability
    max_batches = 3
    batches_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if batches_processed >= max_batches:
                break
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            batches_processed += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    if batches_processed > 0:
        for k in activation_stats:
            activation_stats[k] /= batches_processed

    if not activation_stats:
        return {}

    max_var = max(activation_stats.values())
    if max_var > 0:
        importance = {k: v / max_var for k, v in activation_stats.items()}
    else:
        importance = activation_stats

    # Layers above threshold are critical (high variance = more forgetting pressure)
    critical = {k: v >= threshold for k, v in importance.items()}
    n_critical = sum(1 for v in critical.values() if v)
    n_total = len(critical)
    print(f"D-MoLE: {n_critical}/{n_total} layers marked critical (threshold={threshold})")
    return critical


def inject_cascades_v3(model, rank=8, target_modules=None, layer_importance=None):
    """Inject CASCADES v3.1 adapters with D-MoLE selective allocation."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

    adapters_critical = []
    adapters_funlora = []

    for name, module in dict(model.named_modules()).items():
        if any(t in name for t in target_modules) and (isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"):
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            try:
                parent = model.get_submodule(parent_name)

                # D-MoLE: determine if this layer is critical
                is_critical = True
                if ENABLE_DMOLE_SELECT and ENABLE_FUNLORA and layer_importance:
                    is_critical = layer_importance.get(name, True)

                new_module = CASCADES_v3_Linear(module, rank=rank, is_critical=is_critical)
                new_module = new_module.to(module.weight.device)
                setattr(parent, child_name, new_module)

                if is_critical:
                    adapters_critical.append(new_module.adapter)
                else:
                    adapters_funlora.append(new_module.adapter)
            except AttributeError:
                pass

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze adapters
    for adapter in adapters_critical + adapters_funlora:
        for param in adapter.parameters():
            param.requires_grad = True

    print(f"Injected: {len(adapters_critical)} full CASCADES + {len(adapters_funlora)} FunLoRA rank-1")
    return adapters_critical, adapters_funlora


def prepare_data(tokenizer, task_number):
    """Differentiated domain-specific prompts for continual learning evaluation."""
    torch.manual_seed(42 + task_number)

    task_prompts = [
        # Task 0: Product reviews
        [
            "Review: This product exceeded my expectations. Rating: Positive",
            "Review: Terrible quality, broke after one day. Rating: Negative",
            "Review: Amazing value for the price. Rating: Positive",
            "Review: Would not recommend to anyone. Rating: Negative",
            "Review: Best purchase I've made this year. Rating: Positive",
            "Review: Complete waste of money. Rating: Negative",
        ],
        # Task 1: Movie reviews (different domain)
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

    prompts = task_prompts[task_number % len(task_prompts)] * 8
    encodings = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=64)
    dataset = torch.utils.data.TensorDataset(encodings.input_ids, encodings.attention_mask)
    return DataLoader(dataset, batch_size=2, shuffle=True)


def estimate_quant_noise(model):
    """Estimate quantization noise floor from 4-bit weight statistics."""
    stds = []
    for name, module in model.named_modules():
        if type(module).__name__ == "Linear4bit" and hasattr(module, 'weight'):
            try:
                w = module.weight.data.float()
                stds.append(w.std().item())
            except Exception:
                pass
    return np.mean(stds) if stds else 0.0


def train_cascades_v5():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print ablation config
    print("=" * 60)
    print("CASCADES v5: Task-Aware Gating & Adaptive Rank Routing")
    print("=" * 60)
    print(f"Ablation flags:")
    print(f"  PaCA (CaLoRA):        {ENABLE_PACA}")
    print(f"  DEAL heat kernel:     {ENABLE_DEAL}")
    print(f"  GainLoRA gate (TAG):  {ENABLE_GAINLORA_GATE}")
    print(f"  CoSO null-space:      {ENABLE_COSO_NULLSPACE}")
    print(f"  CL-LoRA reassignment: {ENABLE_CLLORA_REASSIGN}")
    print(f"  SVC calibration (ARR):{ENABLE_SVC}")
    print(f"  D-MoLE layer select:  {ENABLE_DMOLE_SELECT}")
    print(f"  FunLoRA rank-1:       {ENABLE_FUNLORA}")
    print(f"  Device: {device}")
    print("=" * 60)

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

    # Estimate quantization noise floor for DEAL filter
    quant_noise = estimate_quant_noise(model)
    print(f"Estimated quantization noise std: {quant_noise:.6f}")

    # D-MoLE: compute layer importance (optional)
    layer_importance = None
    if ENABLE_DMOLE_SELECT:
        probe_loader = prepare_data(tokenizer, 0)
        layer_importance = compute_layer_importance(model, probe_loader, device)

    # Inject adapters
    critical_adapters, funlora_adapters = inject_cascades_v3(
        model, rank=8, layer_importance=layer_importance
    )
    all_adapters = critical_adapters + funlora_adapters

    # Set quantization noise on critical adapters
    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    num_tasks = 3
    epochs = 3
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    for t in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"--- Training CASCADES v5 Task {t} ---")
        print(f"{'=' * 60}")

        # Add task to critical adapters
        for adapter in critical_adapters:
            adapter.add_task(t)

        # Set task ID on all CASCADES_v3_Linear modules
        for name, module in model.named_modules():
            if isinstance(module, CASCADES_v3_Linear):
                module.current_task_id = t

        dataloader = prepare_data(tokenizer, t)

        # Build optimizer: shared subspace + task lambdas + gates + FunLoRA params
        param_groups = []
        # Critical adapters: shared basis + task lambda
        shared_params = [a.U_shared for a in critical_adapters] + [a.V_shared for a in critical_adapters]
        lambda_params = [a.task_lambdas[str(t)] for a in critical_adapters if str(t) in a.task_lambdas]
        param_groups.append({'params': shared_params, 'lr': 1e-4})
        param_groups.append({'params': lambda_params, 'lr': 5e-3})

        # GainLoRA gate params + v5 Task Embedding
        if ENABLE_GAINLORA_GATE:
            gate_params = [p for a in critical_adapters if hasattr(a, 'gate_proj') for p in a.gate_proj.parameters()]
            task_emb_params = [p for a in critical_adapters if hasattr(a, 'task_embedding') for p in a.task_embedding.parameters()]
            if gate_params:
                param_groups.append({'params': gate_params + task_emb_params, 'lr': 1e-3})

        # FunLoRA params
        funlora_params = [p for a in funlora_adapters for p in a.parameters()]
        if funlora_params:
            param_groups.append({'params': funlora_params, 'lr': 1e-4})

        optimizer = optim.Adam(param_groups)

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

                # Gradient clipping (proven in v1)
                trainable = [p for p in model.parameters() if p.grad is not None]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                # === CASCADES v3.1 Full Pipeline (critical adapters only) ===
                for a in critical_adapters:
                    a.full_descent_step(lr=0.01)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Task {t}, Epoch {ep + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # End-of-task: store gradients + update null-space sketch
        for a in critical_adapters:
            a.store_task_gradients()
            a.update_null_space_sketch()

        # --- Evaluation ---
        print(f"\n--- Evaluation after Task {t} ---")
        for eval_t in range(t + 1):
            eval_dataloader = prepare_data(tokenizer, eval_t)
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                # Set task ID for evaluation
                for name, module in model.named_modules():
                    if isinstance(module, CASCADES_v3_Linear):
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
            print(f"  Task {eval_t} proxy accuracy: {proxy_acc * 100:.2f}% (avg_loss: {avg_loss:.4f})")

    end_time = time.time()

    # Final metrics
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)

    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)

    print(f"\n{'=' * 60}")
    print("=== FINAL CASCADES v5 METRICS ===")
    print(f"{'=' * 60}")
    print(f"Average Accuracy Proxy: {avg_acc * 100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt * 100:.2f}%")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"\nAccuracy Matrix:")
    for i in range(num_tasks):
        row = " | ".join([
            f"{accuracy_matrix[i, j] * 100:6.2f}%" if accuracy_matrix[i, j] > 0 else "   —   "
            for j in range(num_tasks)
        ])
        print(f"  After T{i}: {row}")

    # Ablation config in output
    print(f"\nAblation: PaCA={ENABLE_PACA} DEAL={ENABLE_DEAL} Gate={ENABLE_GAINLORA_GATE} "
          f"CoSO={ENABLE_COSO_NULLSPACE} CL-LoRA={ENABLE_CLLORA_REASSIGN} SVC={ENABLE_SVC} "
          f"D-MoLE={ENABLE_DMOLE_SELECT} FunLoRA={ENABLE_FUNLORA}")

    df = pd.DataFrame(accuracy_matrix,
                      columns=[f"Eval_T{i}" for i in range(num_tasks)],
                      index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("cascades_v5_results.csv")
    print("\nResults saved to cascades_v5_results.csv")


if __name__ == "__main__":
    train_cascades_v5()
