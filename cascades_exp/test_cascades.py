import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import numpy as np
import time

# --- CASCADES Synthetic Empirical Framework ---

class CASCADES_Adapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, svc_lambda=0.01):
        super().__init__()
        self.r = rank
        self.svc_lambda = svc_lambda
        # Shared Live Subspace
        self.U_shared = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(rank))
        self.V_shared = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        
        # Task Specific Coefficients
        self.Lambda_t = None
        self.task_lambdas = nn.ParameterDict()
        
        # EMA Gradient Tracking (GORP)
        self.register_buffer('ema_U', torch.zeros_like(self.U_shared))
        self.register_buffer('ema_V', torch.zeros_like(self.V_shared))
        self.beta1 = 0.9

    def add_task(self, task_id):
        self.task_lambdas[str(task_id)] = nn.Parameter(torch.ones(self.r, self.r))
        self.Lambda_t = self.task_lambdas[str(task_id)]

    def forward(self, x, task_id):
        Lam = self.task_lambdas[str(task_id)]
        # H_t = U_shared * Lambda_t * V_shared * X
        # equivalent linear projection mapping
        proj = x @ self.V_shared.T @ Lam.T @ self.U_shared.T
        return proj
        
    def hamiltonian_descent_step(self, lr=0.01):
        """ Online Subspace Descent keeping parameters on Grassmann manifold organically """
        with torch.no_grad():
            self.ema_U.mul_(self.beta1).add_(self.U_shared.grad, alpha=1 - self.beta1)
            self.ema_V.mul_(self.beta1).add_(self.V_shared.grad, alpha=1 - self.beta1)
            
            # Skew symmetric generators for Stiefel manifold
            G_U = self.ema_U @ self.U_shared.T - self.U_shared @ self.ema_U.T
            G_V = self.ema_V @ self.V_shared.T - self.V_shared @ self.ema_V.T
            
            # Simple Euler approx for exponential map (X_k+1 = exp(-\eta G)X_k)
            self.U_shared.copy_(self.U_shared - lr * (G_U @ self.U_shared))
            self.V_shared.copy_(self.V_shared - lr * (G_V @ self.V_shared))
            
            # SVC Calibration (crush spectral accumulation)
            U, S, V = torch.svd(self.Lambda_t)
            S = S / (1 + self.svc_lambda * S) # calibration
            self.Lambda_t.copy_(U @ torch.diag(S) @ V.T)

class BaseNetwork(nn.Module):
    def __init__(self, in_dim=100, hidden=200, out=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        
        # Freezing core weights as in PEFT
        for p in self.parameters():
            p.requires_grad = False
            
        self.adapter1 = CASCADES_Adapter(in_dim, hidden)
        self.adapter2 = CASCADES_Adapter(hidden, out)

    def add_task(self, task_id):
        self.adapter1.add_task(task_id)
        self.adapter2.add_task(task_id)

    def forward(self, x, task_id):
        h1_base = self.fc1(x)
        h1_adapt = self.adapter1(x, task_id)
        h1 = self.gelu(h1_base + h1_adapt)
        
        out_base = self.fc2(h1)
        out_adapt = self.adapter2(h1, task_id)
        return out_base + out_adapt

# --- Synthetic Data Generation ---
def generate_task_data(num_samples=1000, in_dim=100, classes=10):
    X = torch.randn(num_samples, in_dim)
    # Synthetic target mapping
    W = torch.randn(in_dim, classes)
    y = torch.argmax(X @ W, dim=1)
    return X, y

# --- Training Loop ---
def train_cascades():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    num_tasks = 5
    epochs = 40
    batch_size = 64
    
    model = BaseNetwork().to(device)
    tasks = []
    
    for t in range(num_tasks):
        X_tr, y_tr = generate_task_data()
        X_te, y_te = generate_task_data(300)
        tasks.append({
            'tr': (X_tr.to(device), y_tr.to(device)),
            'te': (X_te.to(device), y_te.to(device))
        })
        
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for t in range(num_tasks):
        print(f"--- Training Task {t} ---")
        model.add_task(t)
        
        # Only optimize the CASCADES shared space and current task's lambdas
        optimizer = optim.Adam([
            {'params': [model.adapter1.U_shared, model.adapter1.V_shared, 
                        model.adapter2.U_shared, model.adapter2.V_shared], 'lr': 1e-4}, # slow shared
            {'params': [model.adapter1.task_lambdas[str(t)], model.adapter2.task_lambdas[str(t)]], 'lr': 5e-3} # fast specialized
        ])
        
        X_tr, y_tr = tasks[t]['tr']
        
        for ep in range(epochs):
            permutation = torch.randperm(X_tr.size()[0])
            for i in range(0, X_tr.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_tr[indices], y_tr[indices]
                
                optimizer.zero_grad()
                out = model(batch_x, t)
                loss = criterion(out, batch_y)
                loss.backward()
                
                # Apply CASCADES custom math instead of standard optimizer step for shared
                model.adapter1.hamiltonian_descent_step()
                model.adapter2.hamiltonian_descent_step()
                
                optimizer.step()
                
        # Eval all tasks up to t
        for eval_t in range(t + 1):
            X_te, y_te = tasks[eval_t]['te']
            with torch.no_grad():
                out = model(X_te, eval_t)
                acc = (torch.argmax(out, dim=1) == y_te).float().mean().item()
                accuracy_matrix[t, eval_t] = acc
                print(f"Performance on Task {eval_t}: {acc*100:.2f}%")

    end_time = time.time()
    
    # Compute Metrics
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)
    
    print("\n--- FINAL CASCADES METRICS ---")
    print(f"Average Accuracy: {avg_acc*100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt*100:.2f}%")
    print(f"Total Computation Time: {end_time - start_time:.2f}s")
    
    df = pd.DataFrame(accuracy_matrix, columns=[f"Eval_T{i}" for i in range(num_tasks)], index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("cascades_exp_results.csv")
    print("Results saved to cascades_exp_results.csv")

if __name__ == "__main__":
    train_cascades()
