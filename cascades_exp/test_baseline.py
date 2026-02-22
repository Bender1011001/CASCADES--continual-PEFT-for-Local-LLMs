import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time

# --- Standard Adapter (No Null Space / SVD Tracking) ---
class Baseline_Adapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.r = rank
        # Standard unshared projections that are updated completely normally
        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(rank))
        self.V = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.task_lambdas = nn.ParameterDict()
        
    def add_task(self, task_id):
        self.task_lambdas[str(task_id)] = nn.Parameter(torch.ones(self.r, self.r))

    def forward(self, x, task_id):
        Lam = self.task_lambdas[str(task_id)]
        proj = x @ self.V.T @ Lam.T @ self.U.T
        return proj

class BaselineNetwork(nn.Module):
    def __init__(self, in_dim=100, hidden=200, out=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        
        for p in self.parameters():
            p.requires_grad = False
            
        self.adapter1 = Baseline_Adapter(in_dim, hidden)
        self.adapter2 = Baseline_Adapter(hidden, out)

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
import math
def generate_task_data(num_samples=1000, in_dim=100, classes=10):
    X = torch.randn(num_samples, in_dim)
    W = torch.randn(in_dim, classes)
    y = torch.argmax(X @ W, dim=1)
    return X, y

# --- Training Loop ---
def train_baseline():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Baseline on {device}")
    
    num_tasks = 5
    epochs = 40
    batch_size = 64
    
    model = BaselineNetwork().to(device)
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
        
        # Standard optimization over all components indiscriminately
        optimizer = optim.Adam([
            {'params': [model.adapter1.U, model.adapter1.V, 
                        model.adapter2.U, model.adapter2.V], 'lr': 1e-4}, 
            {'params': [model.adapter1.task_lambdas[str(t)], model.adapter2.task_lambdas[str(t)]], 'lr': 5e-3}
        ])
        
        X_tr, y_tr = tasks[t]['tr']
        
        for ep in range(epochs):
            permutation = torch.randperm(X_tr.size()[0])
            for i in range(0, X_tr.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_tr[indices], y_tr[indices]
                
                optimizer.zero_grad()
                out = model(batch_x, t) # Note: we use task specific lambda, but base U/V are overwritten
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step() # NO Hamioltonian descent, normal unconstrained steps
                
        # Eval all tasks up to t
        for eval_t in range(t + 1):
            X_te, y_te = tasks[eval_t]['te']
            with torch.no_grad():
                out = model(X_te, eval_t)
                acc = (torch.argmax(out, dim=1) == y_te).float().mean().item()
                accuracy_matrix[t, eval_t] = acc
                print(f"Performance on Task {eval_t}: {acc*100:.2f}%")

    end_time = time.time()
    
    final_accs = accuracy_matrix[-1, :]
    avg_acc = np.mean(final_accs)
    
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(accuracy_matrix[-1, i] - accuracy_matrix[i, i])
    bwt = np.mean(bwt_list)
    
    print("\n--- FINAL BASELINE METRICS ---")
    print(f"Average Accuracy: {avg_acc*100:.2f}%")
    print(f"Backward Transfer (BWT): {bwt*100:.2f}%")
    print(f"Total Computation Time: {end_time - start_time:.2f}s")
    
    df = pd.DataFrame(accuracy_matrix, columns=[f"Eval_T{i}" for i in range(num_tasks)], index=[f"Train_T{i}" for i in range(num_tasks)])
    df.to_csv("baseline_exp_results.csv")
    print("Baseline results saved to baseline_exp_results.csv")

if __name__ == "__main__":
    train_baseline()
