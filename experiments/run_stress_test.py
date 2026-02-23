import os
import subprocess
import numpy as np
import sys

FILE_PATH = "cascades_exp/hf_cascades_reasoning.py"
SEEDS = [42, 123, 999, 2026, 777]

def extract_metrics(output):
    acc, bwt = None, None
    for line in output.split('\n'):
        if "Average Accuracy Proxy:" in line:
            try:
                acc = float(line.split(":")[1].strip().replace('%', ''))
            except ValueError:
                pass
        elif "Backward Transfer (BWT):" in line:
            try:
                bwt = float(line.split(":")[1].strip().replace('%', ''))
            except ValueError:
                pass
    return acc, bwt

def main():
    acc_list = []
    bwt_list = []
    
    python_exe = sys.executable
    
    print(f"Starting CASCADES v9 Stress Test across {len(SEEDS)} seeds: {SEEDS}\n")
    
    for seed in SEEDS:
        print(f"Running Seed {seed}...")
        res = subprocess.run([python_exe, FILE_PATH, "--seed", str(seed)], capture_output=True, text=True)
        acc, bwt = extract_metrics(res.stdout)
        
        if acc is None or bwt is None:
            print(f"Error extracting metrics for seed {seed}")
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
            continue
            
        print(f"  Seed {seed} -> ACC: {acc:.2f}%, BWT: {bwt:+.2f}%")
        acc_list.append(acc)
        bwt_list.append(bwt)
        
    print("\n" + "="*50)
    print("=== FINAL STRESS TEST RESULTS (5 Seeds) ===")
    print("="*50)
    
    if acc_list:
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        bwt_mean = np.mean(bwt_list)
        bwt_std = np.std(bwt_list)
        
        print(f"Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
        print(f"BWT:      {bwt_mean:+.2f}% ± {bwt_std:.2f}%")
    else:
        print("No successful runs to aggregate.")

if __name__ == "__main__":
    main()
