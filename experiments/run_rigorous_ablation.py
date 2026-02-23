import os
import re
import subprocess
import pandas as pd
import sys

FILE_PATH = "cascades_exp/hf_cascades_reasoning.py"
OUTPUT_CSV = "experiments/ablation_matrix.csv"

FLAGS = [
    "ENABLE_PACA",
    "ENABLE_DEAL",
    "ENABLE_GAINLORA_GATE",
    "ENABLE_COSO_NULLSPACE",
    "ENABLE_CLLORA_REASSIGN",
    "ENABLE_SVC",
    "ENABLE_DMOLE_SELECT",
    "ENABLE_FUNLORA"
]

def set_flag(content, flag_name, value):
    val_str = "True" if value else "False"
    pattern = rf"({flag_name}\s*=\s*)(True|False)"
    return re.sub(pattern, rf"\g<1>{val_str}", content, count=1)

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
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        original_content = f.read()

    results = []
    
    python_exe = sys.executable

    try:
        # 1. Run Baseline (All enabled)
        print("Running Baseline (Full CASCADES v9)...")
        content = original_content
        for flag in FLAGS:
            content = set_flag(content, flag, True)
        
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        res = subprocess.run([python_exe, FILE_PATH, "--seed", "42"], capture_output=True, text=True)
        base_acc, base_bwt = extract_metrics(res.stdout)
        
        if base_acc is None or base_bwt is None:
            print("FAILED to extract baseline metrics. STDOUT:")
            print(res.stdout)
            print("STDERR:")
            print(res.stderr)
            return

        print(f"Baseline -> ACC: {base_acc:.2f}%, BWT: {base_bwt:+.2f}%")
        
        results.append({
            "Ablation": "None (Full Pipeline)",
            "ACC (%)": base_acc,
            "BWT (%)": base_bwt,
            "Delta ACC": 0.0,
            "Delta BWT": 0.0
        })

        # 2. Run LOO Ablations
        for flag in FLAGS:
            print(f"\nAblating {flag}...")
            content = original_content
            for f_name in FLAGS:
                content = set_flag(content, f_name, f_name != flag)
            
            with open(FILE_PATH, "w", encoding="utf-8") as f:
                f.write(content)

            res = subprocess.run([python_exe, FILE_PATH, "--seed", "42"], capture_output=True, text=True)
            acc, bwt = extract_metrics(res.stdout)
            
            if acc is None or bwt is None:
                print(f"FAILED to extract metrics for {flag}")
                continue
                
            delta_acc = acc - base_acc
            delta_bwt = bwt - base_bwt
            print(f"{flag} -> ACC: {acc:.2f}% (Δ {delta_acc:+.2f}%), BWT: {bwt:+.2f}% (Δ {delta_bwt:+.2f}%)")
            
            results.append({
                "Ablation": f"- {flag}",
                "ACC (%)": acc,
                "BWT (%)": bwt,
                "Delta ACC": delta_acc,
                "Delta BWT": delta_bwt
            })

    finally:
        # Restore original file
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(original_content)
        print("\nRestored original hf_cascades_reasoning.py")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Ablation matrix saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
