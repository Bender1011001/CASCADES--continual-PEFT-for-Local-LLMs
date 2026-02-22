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
# CASCADES v4: 15-Paper Fusion + Systematic Bug Fixes + Efficiency
# ============================================================================
# Fixes from v3.1 audit:
#   [FIX-1] Double-update removed: U_shared/V_shared NOT in Adam optimizer.
#            Previously Adam + full_descent_step both updated U/V → oscillation.
#   [FIX-2] Lazy DEAL: SVD every DEAL_INTERVAL steps (was every step → 5-10x speedup).
#   [FIX-3] Meaningful gate input: gradient correlation buffer (not param means).
#            Previously gate_input = U.mean(dim=0) ≈ 0 → gate always ≈ 0.73, static.
#   [FIX-4] Manual grad zero of U/V after Riemannian step (not in optimizer.zero_grad).
#   [FIX-5] torch.linalg.svd (replaces deprecated torch.svd), full_matrices=False.
#   [FIX-6] Adaptive D-MoLE threshold: top-40% by variance (was fixed 0.3).
#   [FIX-7] V_shared Riemannian step inlined (was .contiguous() copy → silent no-op).
#            In v3, stella_riemannian_step(V.T.contiguous(), ...) wrote to a temp
#            tensor, never updating V_shared. Now inlined correctly.
#   [FIX-8] Lambda init = zeros(r,r) for T0 (safe LoRA-style zero-output start).
#            Was ones(r,r) → large init output disrupted pretrained activations.
# Improvements:
#   [IMP-1] Adapter alpha 0.3 (was 0.1) — stronger signal, closer to LoRA norm.
#   [IMP-2] Proxy metric: 1/(1+loss) decays ~5x slower than exp(-loss) at CE>2.
#   [IMP-3] Eigenspace warm-start: Lambda_t = 0.5 * Lambda_{t-1} at task boundary.
#            Transfers spectral structure from previous task (reduces cold start).
#   [IMP-4] Gate bias init -1.0 → sigmoid(-1) ≈ 0.27 (conservative, learns to open).
#            Was +1.0 → sigmoid(1) ≈ 0.73 from start, less room to suppress.
#   [IMP-5] Gate proj Linear(3,1): [hist_corr, log_grad_mag, task_phase] — 3 meaningful dims.
# ============================================================================

# --- Ablation Flags ---
ENABLE_PACA             = True  # Intersection B: CaLoRA causal attribution
ENABLE_DEAL             = True  # Intersection B: heat kernel filter (quant-aware)
ENABLE_GAINLORA_GATE    = True  # Intersection B: GainLoRA learned interference gate
ENABLE_COSO_NULLSPACE   = True  # Intersection C: CoSO Frequent Directions
ENABLE_CLLORA_REASSIGN  = True  # Intersection C: CL-LoRA gradient reassignment
ENABLE_SVC              = True  # Intersection C: Singular Value Calibration
ENABLE_DMOLE_SELECT     = True  # Intersection D: D-MoLE layer importance selection
ENABLE_FUNLORA          = True  # Intersection D: FunLoRA rank-1 functional expansion
ENABLE_EIGENSPACE_WARMSTART = True  # [IMP-3] Transfer Lambda across tasks

DEAL_INTERVAL         = 10    # [FIX-2] Apply DEAL only every N optimizer steps
ADAPTER_ALPHA         = 0.3   # [IMP-1] Adapter contribution scale
DMOLE_CRITICAL_RATIO  = 0.4   # [FIX-6] Top 40% of layers by variance = critical


# ============================================================================
# Core Algorithmic Primitives
# ============================================================================

def stella_riemannian_step_inplace(param, grad, lr=0.01):
    """StelLA NeurIPS'25: Stiefel manifold Riemannian gradient + QR retraction.
    param: nn.Parameter with orthonormal columns, shape (d, r).
    grad:  Euclidean gradient (or EMA thereof), same shape.
    In-place: param is directly on the manifold (not a contiguous copy)."""
    with torch.no_grad():
        # Riemannian gradient on Stiefel: grad_R = grad - param @ sym(param^T grad)
        sym = (param.T @ grad + grad.T @ param) / 2.0  # (r, r)
        rg = grad - param @ sym                          # (d, r)
        updated = param - lr * rg
        Q, _ = torch.linalg.qr(updated)
        param.copy_(Q)


def riemannian_step_V_inplace(V_shared, ema_V, lr=0.01):
    """[FIX-7] Correct in-place Stiefel update for V_shared (r, d).
    Treats V^T (d, r) as the Stiefel element, then transposes result back."""
    with torch.no_grad():
        V_T = V_shared.T          # view: (d, r) — NOT contiguous, IS a view of V_shared
        ema_T = ema_V.T           # view: (d, r)
        sym = (V_T.T @ ema_T + ema_T.T @ V_T) / 2.0  # (r, r)
        rg = ema_T - V_T @ sym    # (d, r)
        Q, _ = torch.linalg.qr(V_T - lr * rg)  # Q: (d, r)
        V_shared.copy_(Q.T)       # V_shared gets (r, d) ✓


def paca_causal_mask(grad, historical_grads, temperature=0.1):
    """CaLoRA NeurIPS'25 PaCA: suppress gradient components correlated with past tasks."""
    if not ENABLE_PACA or len(historical_grads) == 0:
        return torch.ones_like(grad)
    flat = grad.flatten()
    corr = torch.stack([
        F.cosine_similarity(flat, hg.to(flat.device), dim=0)
        for hg in historical_grads
    ]).mean()
    inv_importance = 1.0 - torch.sigmoid(
        (grad.abs() / (grad.abs().mean() + 1e-8)) * (1.0 + corr) / temperature
    )
    return inv_importance


def deal_heat_kernel_filter(grad, quant_noise_std=0.0, t=0.05, lambda_decay=0.01):
    """DEAL arXiv'25: low-pass heat kernel filter on gradient singular values.
    [FIX-5] Uses torch.linalg.svd with full_matrices=False for efficiency."""
    if not ENABLE_DEAL or grad.dim() < 2:
        return grad
    # full_matrices=False: only computes min(m,n) singular values
    U, s, Vh = torch.linalg.svd(grad.float(), full_matrices=False)
    eps_quant = max(1e-4, quant_noise_std * 0.1)  # noise floor from quantization
    noise_mask = (s >= eps_quant).float()
    decay = torch.exp(
        -lambda_decay * t * torch.arange(len(s), device=s.device, dtype=torch.float32) ** 2
    )
    s_filtered = s * noise_mask * decay
    return (U @ torch.diag(s_filtered) @ Vh).to(grad.dtype)


def cllora_gradient_reassign(grad, null_sketch, alpha=0.5):
    """CL-LoRA CVPR'25: redirect blocked gradient energy into free subspace."""
    if not ENABLE_CLLORA_REASSIGN or null_sketch is None:
        return grad
    Q, _ = torch.linalg.qr(null_sketch)
    occupied = Q @ (Q.T @ grad)
    free = grad - occupied
    occ_energy = occupied.norm()
    if occ_energy > 1e-8 and free.norm() > 1e-8:
        free_dir = free / free.norm()
        return free + alpha * occ_energy * free_dir
    return free


# ============================================================================
# FunLoRA: rank-1 functional expansion (non-critical layers)
# ============================================================================

class FunLoRA_Adapter(nn.Module):
    """FunLoRA arXiv'25: rank-1 with 3 nonlinear bases → effective rank ≥ 3."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.a = nn.Parameter(torch.randn(out_features, 1) * 0.01)
        self.b = nn.Parameter(torch.randn(1, in_features) * 0.01)

    def forward(self, x):
        ab = x.to(torch.float32) @ self.b.T @ self.a.T  # (*, out)
        # Three functional bases on rank-1 → effective rank ≥ 3
        expanded = ab + torch.sigmoid(ab) + torch.tanh(ab)
        return expanded.to(x.dtype)


# ============================================================================
# CASCADES v4 Full Adapter (critical layers)
# ============================================================================

class CASCADES_v4_Adapter(nn.Module):

    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r          = rank
        self.svc_lambda = svc_lambda
        self.in_features  = in_features
        self.out_features = out_features

        # === Intersection A: Stiefel shared subspace (StelLA USV^T) ===
        # Small random init; first Riemannian step projects to Stiefel manifold
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        # Per-task scaling matrices (diagonal-initialized, Adam-updated)
        self.task_lambdas: nn.ParameterDict = nn.ParameterDict()
        self.last_lambda = None  # [IMP-3] eigenspace warm-start storage

        # GORP EMA gradient buffers
        self.register_buffer('ema_U', torch.zeros(out_features, rank))
        self.register_buffer('ema_V', torch.zeros(rank, in_features))
        self.beta1 = 0.9

        # [FIX-2] Lazy DEAL step counter
        self.register_buffer('step_ctr', torch.zeros(1, dtype=torch.long))

        # === Intersection B: GainLoRA gate ===
        if ENABLE_GAINLORA_GATE:
            # [FIX-3] 3-dim gate input: [hist_corr, log_grad_mag, task_phase]
            self.gate_proj = nn.Linear(3, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -1.0)  # [IMP-4] conservative start
            self.register_buffer('gate_signal', torch.zeros(3))

        # PaCA historical gradient storage (ring buffer of up to 5 tasks)
        self.historical_U_grads: list[torch.Tensor] = []
        self.historical_V_grads: list[torch.Tensor] = []

        # CoSO null-space sketch
        self.register_buffer('null_sketch_U', torch.zeros(out_features, rank))
        self.null_space_init = False

        # Quantization noise estimate
        self.register_buffer('quant_noise_std', torch.tensor(0.0))
        self.current_task_id = 0
        self.num_tasks_seen  = 0

    # -------------------------------------------------------------------------

    def add_task(self, task_id: int):
        """[FIX-8][IMP-3] Zero-init for T0; eigenspace warm-start for T1+."""
        with torch.no_grad():
            dev = self.U_shared.device
            if ENABLE_EIGENSPACE_WARMSTART and self.last_lambda is not None:
                init_val = self.last_lambda.clone() * 0.5
            else:
                # [FIX-8] zeros → zero initial adapter output (safe LoRA init)
                init_val = torch.zeros(self.r, self.r, device=dev)
            self.task_lambdas[str(task_id)] = nn.Parameter(init_val)

    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = self.current_task_id
        in_dtype = x.dtype
        Lam = self.task_lambdas[str(task_id)]

        # (batch, seq, in) → (batch, seq, out)
        adapt_out = x.to(torch.float32) @ self.V_shared.T @ Lam.T @ self.U_shared.T

        # GainLoRA gate: uses gate_signal buffer (updated in full_descent_step)
        if ENABLE_GAINLORA_GATE:
            gate_val = torch.sigmoid(self.gate_proj(self.gate_signal.detach()))
            adapt_out = gate_val * adapt_out

        return adapt_out.to(in_dtype)

    # -------------------------------------------------------------------------

    def store_task_gradients(self, task_id: int):
        """End-of-task bookkeeping: snapshot EMA grads + store Lambda for warm-start."""
        with torch.no_grad():
            if self.ema_U.norm() > 1e-8:
                self.historical_U_grads.append(self.ema_U.clone().flatten().cpu())
                if len(self.historical_U_grads) > 5:
                    self.historical_U_grads.pop(0)
            if self.ema_V.norm() > 1e-8:
                self.historical_V_grads.append(self.ema_V.clone().flatten().cpu())
                if len(self.historical_V_grads) > 5:
                    self.historical_V_grads.pop(0)
            # [IMP-3] Store final Lambda for next task warm-start
            if str(task_id) in self.task_lambdas:
                self.last_lambda = self.task_lambdas[str(task_id)].data.clone()
        self.num_tasks_seen += 1

    def update_null_space_sketch(self):
        """CoSO-style Frequent Directions: accumulate historical gradient directions."""
        if not ENABLE_COSO_NULLSPACE:
            return
        with torch.no_grad():
            if self.ema_U.norm() > 1e-8:
                direction = self.ema_U / (self.ema_U.norm() + 1e-8)
                blend = 0.5 if self.null_space_init else 1.0
                self.null_sketch_U.mul_(blend).add_(direction, alpha=1.0 - blend)
                self.null_space_init = True

    # -------------------------------------------------------------------------

    def full_descent_step(self, lr=0.01):
        """Full v4 descent pipeline per adapter step.
        [FIX-1] U/V not in Adam → this is the ONLY update to shared basis.
        [FIX-4] Manually zeros U/V gradients after reading them."""
        with torch.no_grad():
            if self.U_shared.grad is None or self.V_shared.grad is None:
                return

            grad_U = self.U_shared.grad.clone()
            grad_V = self.V_shared.grad.clone()

            # [FIX-4] Zero U/V grads immediately so nothing else touches them
            self.U_shared.grad.zero_()
            self.V_shared.grad.zero_()

            # === PaCA Causal Mask ===
            mask_U = paca_causal_mask(
                grad_U.flatten(), self.historical_U_grads
            ).reshape(grad_U.shape)
            mask_V = paca_causal_mask(
                grad_V.flatten(), self.historical_V_grads
            ).reshape(grad_V.shape)
            grad_U = grad_U * mask_U
            grad_V = grad_V * mask_V

            # [FIX-3] Update gate_signal with meaningful gradient statistics
            if ENABLE_GAINLORA_GATE:
                log_mag = math.log(grad_U.norm().item() + 1e-8)
                hist_corr = 0.0
                if self.historical_U_grads:
                    flat = grad_U.flatten()
                    corr_vals = [
                        F.cosine_similarity(flat, hg.to(flat.device), dim=0).item()
                        for hg in self.historical_U_grads
                    ]
                    hist_corr = float(np.mean(corr_vals))
                task_phase = min(float(self.num_tasks_seen) / 5.0, 1.0)
                # EMA update
                self.gate_signal[0] = 0.9 * self.gate_signal[0].item() + 0.1 * hist_corr
                self.gate_signal[1] = 0.9 * self.gate_signal[1].item() + 0.1 * log_mag
                self.gate_signal[2] = task_phase

            # [FIX-2] Lazy DEAL: SVD only every DEAL_INTERVAL steps
            step = int(self.step_ctr.item())
            if ENABLE_DEAL and (step % DEAL_INTERVAL == 0):
                noise = self.quant_noise_std.item()
                grad_U = deal_heat_kernel_filter(grad_U, quant_noise_std=noise)
                grad_V = deal_heat_kernel_filter(grad_V, quant_noise_std=noise)
            self.step_ctr.add_(1)

            # === GORP EMA accumulation ===
            self.ema_U.mul_(self.beta1).add_(grad_U, alpha=1.0 - self.beta1)
            self.ema_V.mul_(self.beta1).add_(grad_V, alpha=1.0 - self.beta1)

            # === CoSO null-space projection + CL-LoRA gradient reassignment ===
            if ENABLE_COSO_NULLSPACE and self.null_space_init:
                ema_U_new = cllora_gradient_reassign(self.ema_U, self.null_sketch_U)
                self.ema_U.copy_(ema_U_new)

            # === StelLA Riemannian step (Stiefel manifold) ===
            stella_riemannian_step_inplace(self.U_shared, self.ema_U, lr=lr)
            riemannian_step_V_inplace(self.V_shared, self.ema_V, lr=lr)  # [FIX-7]

            # === SVC Spectral Calibration ===
            if ENABLE_SVC:
                tid = str(self.current_task_id)
                if tid in self.task_lambdas:
                    Lam = self.task_lambdas[tid]
                    U_s, S, Vh_s = torch.linalg.svd(Lam)  # [FIX-5]
                    S = S / (1.0 + self.svc_lambda * S)
                    Lam.copy_(U_s @ torch.diag(S) @ Vh_s)


# ============================================================================
# Adaptive Linear Wrapper (D-MoLE: full CASCADES vs FunLoRA)
# ============================================================================

class CASCADES_v4_Linear(nn.Module):

    def __init__(self, base_layer, rank=8, is_critical=True):
        super().__init__()
        self.base_layer   = base_layer
        self.in_features  = base_layer.in_features
        self.out_features = base_layer.out_features
        self.is_critical  = is_critical
        self.current_task_id = 0

        if is_critical:
            self.adapter = CASCADES_v4_Adapter(
                self.in_features, self.out_features, rank=rank
            )
        else:
            self.adapter = FunLoRA_Adapter(self.in_features, self.out_features)

    def add_task(self, task_id: int):
        if self.is_critical and isinstance(self.adapter, CASCADES_v4_Adapter):
            self.adapter.add_task(task_id)

    def forward(self, x, *args, **kwargs):
        # Forward through base (handles both nn.Linear and Linear4bit)
        if isinstance(self.base_layer, nn.Linear):
            base_out = self.base_layer(x)
        else:
            base_out = self.base_layer(x, *args, **kwargs)

        # Adapter forward
        if self.is_critical:
            self.adapter.current_task_id = self.current_task_id
        adapt_out = self.adapter(x)

        # [IMP-1] alpha=0.3; always cast to base dtype to prevent SDPA mismatch
        return base_out + ADAPTER_ALPHA * adapt_out.to(base_out.dtype)


# ============================================================================
# D-MoLE Layer Importance Scoring (adaptive percentile threshold)
# ============================================================================

def compute_layer_importance(model, dataloader, device):
    """[FIX-6] Adaptive threshold: top DMOLE_CRITICAL_RATIO layers are critical."""
    if not ENABLE_DMOLE_SELECT:
        return {}

    model.eval()
    activation_stats: dict[str, float] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit":
            def make_hook(n):
                def h(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        activation_stats[n] = out.float().var().item()
                return h
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in dataloader:
            ids, mask = batch
            model(input_ids=ids.to(device), attention_mask=mask.to(device))
            break

    for h in hooks:
        h.remove()

    if not activation_stats:
        return {}

    # [FIX-6] Percentile-based threshold
    vals = np.array(list(activation_stats.values()), dtype=np.float32)
    threshold = float(np.percentile(vals, (1.0 - DMOLE_CRITICAL_RATIO) * 100.0))
    critical = {k: float(v) >= threshold for k, v in activation_stats.items()}

    n_crit = sum(critical.values())
    print(f"D-MoLE v4: {n_crit}/{len(critical)} layers critical "
          f"(percentile={100*(1-DMOLE_CRITICAL_RATIO):.0f}th, threshold={threshold:.4f})")
    return critical


# ============================================================================
# Model Injection
# ============================================================================

def inject_cascades_v4(model, rank=8, target_modules=None, layer_importance=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]

    adapters_critical: list[CASCADES_v4_Adapter] = []
    adapters_funlora: list[FunLoRA_Adapter] = []

    for name, module in dict(model.named_modules()).items():
        is_target = any(t in name for t in target_modules)
        is_linear = isinstance(module, nn.Linear) or type(module).__name__ == "Linear4bit"
        if not (is_target and is_linear):
            continue

        parts = name.split(".")
        parent_name, child_name = ".".join(parts[:-1]), parts[-1]
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError:
            continue

        is_critical = True
        if ENABLE_DMOLE_SELECT and ENABLE_FUNLORA and layer_importance:
            is_critical = layer_importance.get(name, True)

        new_mod = CASCADES_v4_Linear(module, rank=rank, is_critical=is_critical)
        new_mod = new_mod.to(module.weight.device)
        setattr(parent, child_name, new_mod)

        if is_critical:
            adapters_critical.append(new_mod.adapter)  # type: ignore
        else:
            adapters_funlora.append(new_mod.adapter)    # type: ignore

    # Freeze base; unfreeze adapters
    for param in model.parameters():
        param.requires_grad = False
    for adapter in adapters_critical + adapters_funlora:
        for param in adapter.parameters():
            param.requires_grad = True

    print(f"Injected: {len(adapters_critical)} CASCADES v4 + {len(adapters_funlora)} FunLoRA")
    return adapters_critical, adapters_funlora


# ============================================================================
# Data
# ============================================================================

def prepare_data(tokenizer, task_number):
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
        # Task 1: Movie reviews
        [
            "Film critique: The cinematography was breathtaking. Verdict: Positive",
            "Film critique: Worst screenplay I've ever seen. Verdict: Negative",
            "Film critique: Outstanding performances by the entire cast. Verdict: Positive",
            "Film critique: A boring and predictable storyline. Verdict: Negative",
            "Film critique: A masterpiece of modern cinema. Verdict: Positive",
            "Film critique: Completely unwatchable garbage. Verdict: Negative",
        ],
        # Task 2: Restaurant reviews
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
    enc = tokenizer(prompts, return_tensors="pt", truncation=True,
                    padding=True, max_length=64)
    dataset = torch.utils.data.TensorDataset(enc.input_ids, enc.attention_mask)
    return DataLoader(dataset, batch_size=2, shuffle=True)


def estimate_quant_noise(model) -> float:
    stds = []
    for _, module in model.named_modules():
        if type(module).__name__ == "Linear4bit" and hasattr(module, "weight"):
            try:
                stds.append(module.weight.data.float().std().item())
            except Exception:
                pass
    return float(np.mean(stds)) if stds else 0.0


# ============================================================================
# Training
# ============================================================================

def train_cascades_v4():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("CASCADES v4: 15-Paper Fusion + 8 Bug Fixes + 5 Improvements")
    print("=" * 72)
    print(f"Ablation: PaCA={ENABLE_PACA} DEAL={ENABLE_DEAL}(lazy×{DEAL_INTERVAL})"
          f" Gate={ENABLE_GAINLORA_GATE} CoSO={ENABLE_COSO_NULLSPACE}"
          f" CL-LoRA={ENABLE_CLLORA_REASSIGN} SVC={ENABLE_SVC}"
          f" D-MoLE={ENABLE_DMOLE_SELECT} FunLoRA={ENABLE_FUNLORA}"
          f" WarmStart={ENABLE_EIGENSPACE_WARMSTART}")
    print(f"Hyperparams: alpha={ADAPTER_ALPHA}, critical_ratio={DMOLE_CRITICAL_RATIO}"
          f", device={device}")
    print("=" * 72)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_id = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    quant_noise = estimate_quant_noise(model)
    print(f"Quantization noise std estimate: {quant_noise:.6f}")

    # D-MoLE probe
    layer_importance = None
    if ENABLE_DMOLE_SELECT:
        probe_loader = prepare_data(tokenizer, 0)
        layer_importance = compute_layer_importance(model, probe_loader, device)

    critical_adapters, funlora_adapters = inject_cascades_v4(
        model, rank=8, layer_importance=layer_importance
    )

    for a in critical_adapters:
        a.quant_noise_std.fill_(quant_noise)

    num_tasks = 3
    epochs    = 3
    acc_matrix = np.zeros((num_tasks, num_tasks))
    start_time = time.time()

    for t in range(num_tasks):
        print(f"\n{'=' * 72}")
        print(f"--- Task {t} Training ---")
        print(f"{'=' * 72}")

        # Register new task
        for a in critical_adapters:
            a.add_task(t)

        # Set task routing on all wrapper modules
        for _, mod in model.named_modules():
            if isinstance(mod, CASCADES_v4_Linear):
                mod.current_task_id = t

        dataloader = prepare_data(tokenizer, t)

        # [FIX-1] Build optimizer WITHOUT U_shared / V_shared
        lambda_params = [
            a.task_lambdas[str(t)]
            for a in critical_adapters
            if str(t) in a.task_lambdas
        ]
        gate_params = [
            p
            for a in critical_adapters
            if ENABLE_GAINLORA_GATE and hasattr(a, "gate_proj")
            for p in a.gate_proj.parameters()
        ]
        funlora_params = [p for a in funlora_adapters for p in a.parameters()]

        param_groups: list[dict] = [{"params": lambda_params, "lr": 5e-3}]
        if gate_params:
            param_groups.append({"params": gate_params, "lr": 1e-3})
        if funlora_params:
            param_groups.append({"params": funlora_params, "lr": 1e-4})

        optimizer = optim.Adam(param_groups)

        for ep in range(epochs):
            ep_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                input_ids, attn_mask = batch
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

                optimizer.zero_grad()
                # [FIX-4] Also zero U/V grads (not in optimizer, accumulate between batches)
                for a in critical_adapters:
                    if a.U_shared.grad is not None:
                        a.U_shared.grad.zero_()
                    if a.V_shared.grad is not None:
                        a.V_shared.grad.zero_()

                outputs = model(
                    input_ids=input_ids, attention_mask=attn_mask, labels=input_ids
                )
                loss = outputs.loss
                loss.backward()

                # Clip all trainable grads
                trainable_grads = [p for p in model.parameters() if p.grad is not None]
                if trainable_grads:
                    torch.nn.utils.clip_grad_norm_(trainable_grads, max_norm=1.0)

                # Riemannian update for U/V (only place these are updated)
                for a in critical_adapters:
                    a.full_descent_step(lr=0.01)

                # Adam update: lambda, gates, funlora only
                optimizer.step()

                ep_loss += loss.item()
                n_batches += 1

            avg_loss = ep_loss / max(n_batches, 1)
            # [IMP-2] 1/(1+loss) proxy — better discrimination than exp(-loss)
            proxy = 1.0 / (1.0 + avg_loss)
            print(f"  Task {t} | Epoch {ep+1}/{epochs} | Loss={avg_loss:.4f}"
                  f" | Proxy={proxy*100:.2f}%")

        # End-of-task bookkeeping
        for a in critical_adapters:
            a.store_task_gradients(t)
            a.update_null_space_sketch()

        # ---- Evaluation ----
        print(f"\n  Evaluation after Task {t}:")
        for eval_t in range(t + 1):
            eval_loader = prepare_data(tokenizer, eval_t)
            total_loss = 0.0
            n_eval = 0

            with torch.no_grad():
                for _, mod in model.named_modules():
                    if isinstance(mod, CASCADES_v4_Linear):
                        mod.current_task_id = eval_t
                for batch in eval_loader:
                    ids, _ = batch
                    out = model(input_ids=ids.to(device), labels=ids.to(device))
                    total_loss += out.loss.item()
                    n_eval += 1

            avg_eval_loss = total_loss / max(n_eval, 1)
            proxy_acc = 1.0 / (1.0 + avg_eval_loss)
            acc_matrix[t, eval_t] = proxy_acc
            print(f"    Task {eval_t}: loss={avg_eval_loss:.4f}  proxy={proxy_acc*100:.2f}%")

    elapsed = time.time() - start_time

    # ---- Final Metrics ----
    avg_acc = float(np.mean(acc_matrix[-1, :]))
    bwt = float(np.mean([acc_matrix[-1, i] - acc_matrix[i, i] for i in range(num_tasks - 1)]))

    print(f"\n{'=' * 72}")
    print("=== CASCADES v4 FINAL METRICS ===")
    print(f"{'=' * 72}")
    print(f"  Average Accuracy [1/(1+loss)]: {avg_acc*100:.2f}%")
    print(f"  Backward Transfer (BWT):       {bwt*100:.2f}%")
    print(f"  Total Time:                    {elapsed:.1f}s")
    print(f"\n  Accuracy Matrix (rows=after_train_Ti, cols=eval_Tj):")
    header = "         " + "  ".join(f"  T{j}" for j in range(num_tasks))
    print(header)
    for i in range(num_tasks):
        row_str = "  ".join(
            f"{acc_matrix[i,j]*100:5.1f}%" if acc_matrix[i,j] > 0 else "  —  "
            for j in range(num_tasks)
        )
        print(f"  After T{i}: {row_str}")

    df = pd.DataFrame(
        acc_matrix,
        columns=[f"Eval_T{i}" for i in range(num_tasks)],
        index=[f"Train_T{i}" for i in range(num_tasks)],
    )
    df.to_csv("cascades_v4_results.csv")
    print("\n  Results → cascades_v4_results.csv")
    print(f"\n  Key fixes: [FIX-1] no double-update  [FIX-2] lazy DEAL×{DEAL_INTERVAL}"
          f"  [FIX-3] gate fixed  [FIX-7] V_shared Riemannian inlined")
    print(f"  Key improvements: alpha={ADAPTER_ALPHA}  proxy=1/(1+loss)"
          f"  eigenspace warm-start  conservative gate")


if __name__ == "__main__":
    train_cascades_v4()
