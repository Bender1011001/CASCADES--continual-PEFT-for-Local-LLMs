"""
Ablation study runner.

Runs the full ablation table from the paper (Appendix B) by systematically
disabling one CASCADES component at a time and recording ACC / BWT.

Usage
-----
python experiments/ablation.py --seeds 0 --model tinyllama  # fast smoke test
python experiments/ablation.py --seeds 0 1 2               # full study (hours)

Output: results/ablation_<timestamp>.csv
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Ablation conditions: (label, extra CLI flags)
ABLATION_CONDITIONS = [
    ("Full CASCADES v5",    []),
    ("No PaCA",             ["--no-paca"]),
    ("No DEAL",             ["--no-deal"]),
    ("No Gate",             ["--no-gate"]),
    ("No CoSO",             ["--no-coso"]),
    ("No EAR",              ["--no-ear"]),
    ("No SVC",              ["--no-svc"]),
    ("No D-MoLE",           ["--no-dmole"]),
    ("No FunLoRA",          ["--no-funlora"]),
    ("LoRA Baseline",       ["--method", "lora"]),
]


def run_condition(label: str, extra_flags: list[str], seeds: list[str], model: str) -> dict:
    """Invoke run_experiment.py as a subprocess and parse its output."""
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_experiment.py"),
        "--method", "cascades_v5",
        "--model", model,
        "--seeds", *seeds,
        *extra_flags,
    ]
    print(f"\n{'='*60}")
    print(f"  Ablation: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    return {
        "condition": label,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run full CASCADES ablation study")
    p.add_argument("--seeds", nargs="+", default=["0"])
    p.add_argument("--model", default="qwen3_4b")
    args = p.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = ROOT / "results" / f"ablation_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, extra_flags in ABLATION_CONDITIONS:
        row = run_condition(label, extra_flags, args.seeds, args.model)
        rows.append(row)
        print(f"  Done ({row['elapsed_s']:.1f}s, rc={row['returncode']})")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "elapsed_s", "returncode"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nAblation log saved → {out_path}")
    print("(ACC/BWT values are in results/summary.csv — merge on timestamp)")


if __name__ == "__main__":
    main()
