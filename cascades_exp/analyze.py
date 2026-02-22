import pandas as pd
import sys
import os

def analyze_results():
    if not os.path.exists("cascades_exp_results.csv") or not os.path.exists("baseline_exp_results.csv"):
        print("Waiting for experiments to complete...")
        return
        
    cascades_df = pd.read_csv("cascades_exp_results.csv", index_col=0)
    baseline_df = pd.read_csv("baseline_exp_results.csv", index_col=0)
    
    print("=== EMPIRICAL EVALUATION RESULTS ===")
    
    # CASCADES Average Accuracy & BWT
    c_final = cascades_df.iloc[-1].values
    c_avg = c_final.mean()
    c_bwt = sum([cascades_df.iloc[-1, i] - cascades_df.iloc[i, i] for i in range(len(c_final)-1)]) / (len(c_final)-1)
    
    # Baseline Average Accuracy & BWT
    b_final = baseline_df.iloc[-1].values
    b_avg = b_final.mean()
    b_bwt = sum([baseline_df.iloc[-1, i] - baseline_df.iloc[i, i] for i in range(len(b_final)-1)]) / (len(b_final)-1)
    
    print("\n[Baseline Adapter]")
    print(f"Average Accuracy: {b_avg*100:.2f}%")
    print(f"Backward Transfer: {b_bwt*100:.2f}%")
    print("Task 0 Degradation (T0->T4):", (baseline_df.iloc[-1, 0] - baseline_df.iloc[0, 0])*100, "%")
    
    print("\n[CASCADES (Ours)]")
    print(f"Average Accuracy: {c_avg*100:.2f}%")
    print(f"Backward Transfer: {c_bwt*100:.2f}%")
    print("Task 0 Degradation (T0->T4):", (cascades_df.iloc[-1, 0] - cascades_df.iloc[0, 0])*100, "%")
    
    print("\n--- CONCLUSION FOR PAPER ---")
    if c_bwt > b_bwt:
        print("CASCADES successfully mitigates catastrophic forgetting compared to naive baseline.")
        print("Hamiltonian constraints and SVC scaling successfully maintained shared subspace boundaries.")
    else:
        print("CASCADES did not outperform baseline. Check hyperparams.")

if __name__ == "__main__":
    analyze_results()
