from __future__ import annotations

import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import cascades.data as data
import experiment_matrix
import research_runner
import train


CURRENT4 = list(data.TASK_FILES)
REASONING3 = list(experiment_matrix.DEFAULT_TASK_FILES)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"unavailable: {exc}"


def count_examples(path: str) -> int | None:
    p = ROOT / path
    if not p.exists():
        return None
    return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())


def task_manifest(files: list[str]) -> list[dict]:
    rows = []
    for i, file_name in enumerate(files):
        p = ROOT / file_name
        rows.append(
            {
                "task_index": i,
                "path": file_name,
                "exists": p.exists(),
                "examples": count_examples(file_name),
                "bytes": p.stat().st_size if p.exists() else None,
            }
        )
    return rows


def runner_forwarding_snapshot() -> dict:
    src = inspect.getsource(research_runner.ExperimentRunner._run_cascades)
    return {
        "forwards_rank": '"rank"' in src or "'rank'" in src,
        "forwards_max_length": "max_length" in src,
        "pops_task_files": 'task_files = overrides.pop("task_files", None)' in src,
        "calls_train_cascades_kwargs": "accuracy_matrix = train_cascades(**kwargs)" in src,
    }


def num_samples_snapshot() -> dict:
    text = (ROOT / "train.py").read_text(encoding="utf-8-sig")
    positions = [m.start() for m in re.finditer("num_samples", text)]
    prepare_data_lines = [
        line.strip() for line in text.splitlines() if "prepare_data(" in line
    ]
    return {
        "occurrence_count": len(positions),
        "positions": positions,
        "prepare_data_call_lines": prepare_data_lines,
        "wired_to_prepare_data": "num_samples" in "\n".join(prepare_data_lines),
    }


def main() -> None:
    out_dir = ROOT / "experiments" / "cf_cycle_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "current_num_tasks": data.NUM_TASKS,
        "current_task_files": CURRENT4,
        "reasoning3_task_files": REASONING3,
        "current4_manifest": task_manifest(CURRENT4),
        "reasoning3_manifest": task_manifest(REASONING3),
        "train_cascades_signature": str(inspect.signature(train.train_cascades)),
        "runner_forwarding": runner_forwarding_snapshot(),
        "num_samples": num_samples_snapshot(),
        "gpu_preconditions": {
            "ok": False,
            "reason": "CF-cycle-1 code-mode task generated CPU-light audit/runner artifacts only; launch GPU ablation explicitly after reviewing this snapshot.",
        },
    }

    (out_dir / "audit_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    (out_dir / "task_manifest_current4.json").write_text(
        json.dumps(snapshot["current4_manifest"], indent=2), encoding="utf-8"
    )
    (out_dir / "task_manifest_reasoning3.json").write_text(
        json.dumps(snapshot["reasoning3_manifest"], indent=2), encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
