import os
import subprocess
import re
import time

target_script = "cascades_exp/hf_cascades_reasoning.py"

configs = [
    {"name": "Full_CASCADES_v8", "flags": {}},
    {"name": "No_DMoLE", "flags": {"ENABLE_DMOLE_SELECT": "False", "ENABLE_FUNLORA": "False"}},
    {"name": "No_Autopoiesis_EAR", "flags": {"ENABLE_CLLORA_REASSIGN": "False", "ENABLE_COSO_NULLSPACE": "False"}},
    {"name": "Baseline_LoRA", "flags": {
        "ENABLE_PACA": "False",
        "ENABLE_DEAL": "False",
        "ENABLE_GAINLORA_GATE": "False",
        "ENABLE_COSO_NULLSPACE": "False",
        "ENABLE_CLLORA_REASSIGN": "False",
        "ENABLE_SVC": "False",
        "ENABLE_DMOLE_SELECT": "False",
        "ENABLE_FUNLORA": "False"
    }}
]

def update_flags(flags):
    with open(target_script, "r", encoding="utf-8") as f:
        content = f.read()
    
    # reset all to True first
    all_flags = ["ENABLE_PACA", "ENABLE_DEAL", "ENABLE_GAINLORA_GATE", 
                 "ENABLE_COSO_NULLSPACE", "ENABLE_CLLORA_REASSIGN", "ENABLE_SVC", 
                 "ENABLE_DMOLE_SELECT", "ENABLE_FUNLORA"]
    for flag in all_flags:
        content = re.sub(rf"{flag}\s*=\s*(True|False)", f"{flag} = True", content)
        
    for flag, val in flags.items():
        content = re.sub(rf"{flag}\s*=\s*(True|False)", f"{flag} = {val}", content)
        
    with open(target_script, "w", encoding="utf-8") as f:
        f.write(content)

def extract_metrics(log_content):
    metrics = []
    capture = False
    for line in log_content.split("\n"):
        if "FINAL CASCADES REASONING METRICS" in line:
            capture = True
        if capture:
            metrics.append(line)
            if "Results saved" in line:
                break
    return "\n".join(metrics)

with open("cascades_final_benchmark_suite.log", "w", encoding="utf-8") as log_file:
    log_file.write("Starting Final CASCADES v8 Benchmark Suite...\n")

for config in configs:
    name = config["name"]
    flags = config["flags"]
    print(f"Running ablation: {name}")
    
    update_flags(flags)
    
    start_time = time.time()
    result = subprocess.run(["python", "-u", target_script], capture_output=True, text=True, errors="replace")
    run_time = time.time() - start_time
    
    # Save output to individual log
    with open(f"cascades_run_{name}.log", "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write(result.stderr)
        
    metrics_str = extract_metrics(result.stdout)
    if not metrics_str:
        metrics_str = "Error: Metrics not found. Check individual run log."
        
    with open("cascades_final_benchmark_suite.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Results for {name} (Time: {run_time:.2f}s):\n")
        log_file.write(metrics_str + "\n")
    print(f"Finished {name} in {run_time:.2f}s")

# Reset back to full
update_flags({})
print("Benchmarking completely automatically finished. Suite saved to cascades_final_benchmark_suite.log.")
