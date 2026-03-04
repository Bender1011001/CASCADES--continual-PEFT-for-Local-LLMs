"""
CASCADES Research Loop — Post-Run Analysis.

Reads experiments/results.csv, computes deltas vs baseline, ranks component
contributions, and generates a publication-quality markdown analysis report.

Usage:
    python research_analyzer.py                          # Generate report
    python research_analyzer.py --format latex            # LaTeX tables
    python research_analyzer.py --csv path/to/results.csv # Custom CSV path
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Component names for ablation analysis
# ---------------------------------------------------------------------------

# Maps experiment ID to the component being ablated (Cycle 3)
ABLATION_MAP = {
    "3.1": "EAR / Null-space",
    "3.2": "Sleep Consolidation",
    "3.3": "PaCA (Causal Attribution)",
    "3.4": "Breathing / SVC",
    "3.5": "DEAL Filter",
}

# Maps experiment ID to the v10 patch being tested (Cycle 2)
V10_PATCH_MAP = {
    "2.1": "Frozen Null-space",
    "2.2": "Soft-EAR",
    "2.3": "GQA Preconditioning",
    "2.4": "Principal Expansion",
    "2.5": "CFG Decoding",
    "2.6": "All v10 Combined",
}

# Cycle names
CYCLE_NAMES = {
    1: "Baselines",
    2: "v10 Patch Validation",
    3: "Component Ablations",
    4: "Generative Gap Investigation",
    5: "Scaling",
}


# ---------------------------------------------------------------------------
# ResearchAnalyzer
# ---------------------------------------------------------------------------

class ResearchAnalyzer:
    """Analyze CASCADES experiment results from the research loop CSV.

    Provides methods for computing deltas, ranking components, and
    generating publication-quality markdown reports.
    """

    def __init__(self, results_csv: str = "experiments/results.csv"):
        self.csv_path = Path(results_csv)
        self.df: Optional[pd.DataFrame] = None
        self.baseline_id = "1.1"   # Plain LoRA
        self.v9_baseline_id = "1.2"  # CASCADES v9 reproduction

    def load_results(self) -> pd.DataFrame:
        """Load results from CSV into a DataFrame.

        Returns:
            DataFrame with all experiment results.

        Raises:
            FileNotFoundError: If results CSV doesn't exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.csv_path}\n"
                f"Run experiments first: python research_runner.py"
            )

        self.df = pd.read_csv(self.csv_path)

        # Convert numeric columns
        numeric_cols = [
            "t0_acc", "t1_acc", "t2_acc", "t3_acc", "t4_acc",
            "avg_acc", "bwt", "fwt",
            "em_exact", "em_normalized", "em_containment",
            "vram_peak_mb", "wall_time_s", "cfg_lambda", "rank",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Filter to completed experiments only
        if "status" in self.df.columns:
            completed = self.df[self.df["status"] == "completed"]
            print(f"Loaded {len(completed)} completed experiments "
                  f"(of {len(self.df)} total)")
            self.df = completed.reset_index(drop=True)
        else:
            print(f"Loaded {len(self.df)} experiments")

        return self.df

    def compute_deltas(self, baseline_id: str = "1.1") -> pd.DataFrame:
        """Compute ACC/BWT deltas vs the LoRA baseline for every experiment.

        Args:
            baseline_id: Experiment ID to use as baseline (default "1.1").

        Returns:
            DataFrame with added delta_acc and delta_bwt columns.
        """
        if self.df is None:
            self.load_results()

        baseline_rows = self.df[self.df["experiment_id"] == baseline_id]
        if baseline_rows.empty:
            print(f"Warning: baseline {baseline_id} not found in results")
            deltas = self.df.copy()
            deltas["delta_acc"] = np.nan
            deltas["delta_bwt"] = np.nan
            return deltas

        baseline = baseline_rows.iloc[0]
        deltas = self.df.copy()
        deltas["delta_acc"] = deltas["avg_acc"] - baseline["avg_acc"]
        deltas["delta_bwt"] = deltas["bwt"] - baseline["bwt"]
        return deltas

    def component_ranking(self) -> list[dict]:
        """Rank CASCADES components by their ACC contribution.

        Computes how much ACC drops when each component is removed
        (Cycle 3 ablations vs v9 baseline 1.2).

        Returns:
            Sorted list of dicts: [{"component", "delta_acc", "delta_bwt", "verdict"}]
        """
        if self.df is None:
            self.load_results()

        # Get v9 baseline
        v9_rows = self.df[self.df["experiment_id"] == self.v9_baseline_id]
        if v9_rows.empty:
            print("Warning: v9 baseline (1.2) not found — cannot compute rankings")
            return []

        v9 = v9_rows.iloc[0]
        rankings = []

        for exp_id, component_name in ABLATION_MAP.items():
            exp_rows = self.df[self.df["experiment_id"] == exp_id]
            if exp_rows.empty:
                continue

            exp = exp_rows.iloc[0]
            delta_acc = exp["avg_acc"] - v9["avg_acc"]
            delta_bwt = exp["bwt"] - v9["bwt"]

            # Classify impact
            if abs(delta_acc) > 0.05 or abs(delta_bwt) > 0.05:
                verdict = "Critical"
            elif abs(delta_acc) > 0.02 or abs(delta_bwt) > 0.02:
                verdict = "Important"
            elif abs(delta_acc) > 0.01 or abs(delta_bwt) > 0.01:
                verdict = "Moderate"
            else:
                verdict = "Minor"

            rankings.append({
                "component": component_name,
                "experiment_id": exp_id,
                "delta_acc": delta_acc,
                "delta_bwt": delta_bwt,
                "abs_impact": abs(delta_acc) + abs(delta_bwt),
                "verdict": verdict,
            })

        # Sort by absolute impact (most impactful first)
        rankings.sort(key=lambda x: x["abs_impact"], reverse=True)
        return rankings

    def _format_pct(self, value: float, show_sign: bool = True) -> str:
        """Format a float as a percentage string."""
        if pd.isna(value):
            return "N/A"
        if show_sign:
            return f"{value*100:+.2f}%"
        return f"{value*100:.2f}%"

    def _component_contribution_table(self) -> str:
        """Generate markdown table: component → ACC/BWT contribution."""
        rankings = self.component_ranking()
        if not rankings:
            return "*No ablation data available yet.*\n"

        lines = [
            "| Component | ACC Δ vs v9 | BWT Δ vs v9 | Verdict |",
            "|-----------|-------------|-------------|---------|",
        ]
        for r in rankings:
            lines.append(
                f"| {r['component']:<25s} | {self._format_pct(r['delta_acc']):>11s} "
                f"| {self._format_pct(r['delta_bwt']):>11s} "
                f"| {r['verdict']:<9s} |"
            )
        return "\n".join(lines) + "\n"

    def _v10_patch_table(self) -> str:
        """Generate markdown table: v10 patch → marginal improvement over v9."""
        if self.df is None:
            self.load_results()

        v9_rows = self.df[self.df["experiment_id"] == self.v9_baseline_id]
        if v9_rows.empty:
            return "*v9 baseline (1.2) not found.*\n"

        v9 = v9_rows.iloc[0]

        lines = [
            "| v10 Patch | ACC Δ vs v9 | BWT Δ vs v9 | VRAM (MB) | Verdict |",
            "|-----------|-------------|-------------|-----------|---------|",
        ]

        for exp_id, patch_name in V10_PATCH_MAP.items():
            exp_rows = self.df[self.df["experiment_id"] == exp_id]
            if exp_rows.empty:
                continue

            exp = exp_rows.iloc[0]
            delta_acc = exp["avg_acc"] - v9["avg_acc"]
            delta_bwt = exp["bwt"] - v9["bwt"]
            vram = exp.get("vram_peak_mb", 0)

            # Verdict
            if delta_acc > 0.01 or delta_bwt > 0.01:
                verdict = "Keep"
            elif abs(delta_acc) < 0.005 and abs(delta_bwt) < 0.005:
                verdict = "Skip"
            else:
                verdict = "Marginal"

            lines.append(
                f"| {patch_name:<22s} | {self._format_pct(delta_acc):>11s} "
                f"| {self._format_pct(delta_bwt):>11s} "
                f"| {vram:>9.0f} | {verdict:<7s} |"
            )

        return "\n".join(lines) + "\n"

    def _generative_gap_table(self) -> str:
        """Generate markdown table: eval strategy → EM/containment scores."""
        if self.df is None:
            self.load_results()

        cycle4 = self.df[self.df["cycle"] == 4]
        if cycle4.empty:
            return "*No Cycle 4 (generative gap) data available.*\n"

        lines = [
            "| Strategy | EM Exact | EM Normalized | Containment | Proxy ACC |",
            "|----------|----------|---------------|-------------|-----------|",
        ]

        # Include v9 baseline for comparison
        for exp_id in [self.v9_baseline_id, "4.1", "4.2", "4.3"]:
            exp_rows = self.df[self.df["experiment_id"] == exp_id]
            if exp_rows.empty:
                continue

            exp = exp_rows.iloc[0]
            name = exp.get("experiment_name", exp_id)

            lines.append(
                f"| {name:<24s} "
                f"| {self._format_pct(exp.get('em_exact', 0), False):>8s} "
                f"| {self._format_pct(exp.get('em_normalized', 0), False):>13s} "
                f"| {self._format_pct(exp.get('em_containment', 0), False):>11s} "
                f"| {self._format_pct(exp.get('avg_acc', 0), False):>9s} |"
            )

        return "\n".join(lines) + "\n"

    def _overview_table(self) -> str:
        """Generate full results overview table."""
        if self.df is None:
            self.load_results()

        deltas = self.compute_deltas()

        lines = [
            "| ID | Name | Cycle | ACC | BWT | Δ ACC | Δ BWT | VRAM (MB) | Time (s) |",
            "|----|------|-------|-----|-----|-------|-------|-----------|----------|",
        ]

        for _, row in deltas.iterrows():
            lines.append(
                f"| {row['experiment_id']:>4s} "
                f"| {str(row.get('experiment_name', '')):<28s} "
                f"| {int(row.get('cycle', 0)):>5d} "
                f"| {self._format_pct(row.get('avg_acc', 0), False):>5s} "
                f"| {self._format_pct(row.get('bwt', 0)):>5s} "
                f"| {self._format_pct(row.get('delta_acc', 0)):>7s} "
                f"| {self._format_pct(row.get('delta_bwt', 0)):>7s} "
                f"| {row.get('vram_peak_mb', 0):>9.0f} "
                f"| {row.get('wall_time_s', 0):>8.0f} |"
            )

        return "\n".join(lines) + "\n"

    def _flag_regressions(self) -> list[str]:
        """Return list of experiments where BWT < LoRA baseline."""
        if self.df is None:
            self.load_results()

        baseline_rows = self.df[self.df["experiment_id"] == self.baseline_id]
        if baseline_rows.empty:
            return []

        baseline_bwt = baseline_rows.iloc[0]["bwt"]
        regressions = []

        for _, row in self.df.iterrows():
            if row["experiment_id"] == self.baseline_id:
                continue
            if pd.notna(row.get("bwt")) and row["bwt"] < baseline_bwt:
                regressions.append(
                    f"⚠ {row['experiment_id']} ({row.get('experiment_name', '')}): "
                    f"BWT={self._format_pct(row['bwt'])} < baseline {self._format_pct(baseline_bwt)}"
                )

        return regressions

    def _key_findings(self) -> list[str]:
        """Extract key findings from the results."""
        if self.df is None:
            self.load_results()

        findings = []

        # Best ACC
        if not self.df.empty and "avg_acc" in self.df.columns:
            best_idx = self.df["avg_acc"].idxmax()
            if pd.notna(best_idx):
                best = self.df.loc[best_idx]
                findings.append(
                    f"**Best ACC**: {best.get('experiment_name', best['experiment_id'])} "
                    f"at {self._format_pct(best['avg_acc'], False)}"
                )

        # Best BWT
        if not self.df.empty and "bwt" in self.df.columns:
            best_bwt_idx = self.df["bwt"].idxmax()
            if pd.notna(best_bwt_idx):
                best_bwt = self.df.loc[best_bwt_idx]
                findings.append(
                    f"**Best BWT**: {best_bwt.get('experiment_name', best_bwt['experiment_id'])} "
                    f"at {self._format_pct(best_bwt['bwt'])}"
                )

        # v9 vs baseline improvement
        baseline_rows = self.df[self.df["experiment_id"] == self.baseline_id]
        v9_rows = self.df[self.df["experiment_id"] == self.v9_baseline_id]
        if not baseline_rows.empty and not v9_rows.empty:
            b = baseline_rows.iloc[0]
            v = v9_rows.iloc[0]
            delta = v["avg_acc"] - b["avg_acc"]
            findings.append(
                f"**CASCADES v9 vs LoRA**: "
                f"{self._format_pct(delta)} ACC improvement"
            )

        # Component rankings
        rankings = self.component_ranking()
        if rankings:
            top = rankings[0]
            findings.append(
                f"**Most critical component**: {top['component']} "
                f"(removing it: ACC {self._format_pct(top['delta_acc'])})"
            )

        # Regressions
        regressions = self._flag_regressions()
        if regressions:
            findings.append(f"**Regressions detected**: {len(regressions)} experiments")
        else:
            findings.append("**No BWT regressions** detected vs LoRA baseline")

        return findings

    def generate_report(self) -> str:
        """Generate full analysis report as markdown string.

        Returns:
            Complete markdown report with all tables and findings.
        """
        if self.df is None:
            self.load_results()

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        n_completed = len(self.df)

        sections = []

        # Header
        sections.append(f"# CASCADES Research Loop — Analysis Report\n")
        sections.append(f"Generated: {timestamp}  ")
        sections.append(f"Experiments completed: {n_completed}/18\n")

        # Key findings
        sections.append("## Key Findings\n")
        findings = self._key_findings()
        for f in findings:
            sections.append(f"- {f}")
        sections.append("")

        # Full overview
        sections.append("## Full Results Overview\n")
        sections.append(self._overview_table())

        # Per-cycle analysis
        for cycle_num in range(1, 6):
            cycle_data = self.df[self.df["cycle"] == cycle_num]
            if cycle_data.empty:
                continue

            cycle_name = CYCLE_NAMES.get(cycle_num, f"Cycle {cycle_num}")
            sections.append(f"## Cycle {cycle_num}: {cycle_name}\n")

            if cycle_num == 2:
                sections.append("### v10 Patch Impact\n")
                sections.append(self._v10_patch_table())

            elif cycle_num == 3:
                sections.append("### Component Contribution Ranking\n")
                sections.append(self._component_contribution_table())

            elif cycle_num == 4:
                sections.append("### Generative Gap Analysis\n")
                sections.append(self._generative_gap_table())

            # Per-cycle summary stats
            if not cycle_data.empty and "avg_acc" in cycle_data.columns:
                mean_acc = cycle_data["avg_acc"].mean()
                mean_bwt = cycle_data["bwt"].mean()
                sections.append(
                    f"*Cycle {cycle_num} averages: "
                    f"ACC={self._format_pct(mean_acc, False)}, "
                    f"BWT={self._format_pct(mean_bwt)}*\n"
                )

        # Regressions
        regressions = self._flag_regressions()
        if regressions:
            sections.append("## ⚠ Regressions\n")
            for r in regressions:
                sections.append(f"- {r}")
            sections.append("")

        # Resource usage
        sections.append("## Resource Usage\n")
        if "vram_peak_mb" in self.df.columns:
            max_vram = self.df["vram_peak_mb"].max()
            mean_vram = self.df["vram_peak_mb"].mean()
            sections.append(f"- Peak VRAM: {max_vram:.0f} MB (mean: {mean_vram:.0f} MB)")
        if "wall_time_s" in self.df.columns:
            total_time = self.df["wall_time_s"].sum()
            mean_time = self.df["wall_time_s"].mean()
            sections.append(
                f"- Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)"
            )
            sections.append(f"- Mean per experiment: {mean_time:.0f}s")
        sections.append("")

        # Methodology note
        sections.append("## Methodology\n")
        sections.append(
            "All experiments use Qwen3-4B NF4 quantized (~5.2GB VRAM) "
            "on RTX 4060 Ti 8GB. Proxy accuracy = exp(-avg_cross_entropy_loss). "
            "BWT = mean(A[T-1,i] - A[i,i]) for i < T-1. "
            "Baseline (1.1) uses standard PEFT LoRA(r=8, α=16) with "
            "identical data pipeline and evaluation.\n"
        )

        return "\n".join(sections)

    def save_report(self, path: str = "experiments/analysis_report.md") -> None:
        """Write the analysis report to a file.

        Args:
            path: Output file path (default: experiments/analysis_report.md).
        """
        report = self.generate_report()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    def generate_latex_tables(self) -> str:
        """Generate LaTeX-formatted tables for paper inclusion.

        Returns:
            LaTeX string with tabular environments.
        """
        if self.df is None:
            self.load_results()

        sections = []

        # Main results table
        sections.append("% Main Results Table")
        sections.append("\\begin{table}[ht]")
        sections.append("\\centering")
        sections.append("\\caption{CASCADES Experiment Results}")
        sections.append("\\label{tab:results}")
        sections.append("\\begin{tabular}{llrrrrr}")
        sections.append("\\toprule")
        sections.append("ID & Name & ACC (\\%) & BWT (\\%) & $\\Delta$ ACC & VRAM (MB) & Time (s) \\\\")
        sections.append("\\midrule")

        for _, row in self.df.iterrows():
            name = str(row.get("experiment_name", ""))[:25]
            acc = row.get("avg_acc", 0) * 100
            bwt_val = row.get("bwt", 0) * 100
            vram = row.get("vram_peak_mb", 0)
            wtime = row.get("wall_time_s", 0)

            # Delta vs baseline
            baseline_rows = self.df[self.df["experiment_id"] == self.baseline_id]
            if not baseline_rows.empty:
                delta = (row.get("avg_acc", 0) - baseline_rows.iloc[0]["avg_acc"]) * 100
            else:
                delta = 0.0

            sections.append(
                f"{row['experiment_id']} & {name} & "
                f"{acc:.1f} & {bwt_val:+.1f} & {delta:+.1f} & "
                f"{vram:.0f} & {wtime:.0f} \\\\"
            )

        sections.append("\\bottomrule")
        sections.append("\\end{tabular}")
        sections.append("\\end{table}")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CASCADES Research Loop — Analysis Report Generator",
    )
    parser.add_argument(
        "--csv", type=str, default="experiments/results.csv",
        help="Path to results CSV (default: experiments/results.csv)",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/analysis_report.md",
        help="Output report path (default: experiments/analysis_report.md)",
    )
    parser.add_argument(
        "--format", type=str, choices=["markdown", "latex"], default="markdown",
        help="Output format (default: markdown)",
    )
    args = parser.parse_args()

    analyzer = ResearchAnalyzer(results_csv=args.csv)

    try:
        analyzer.load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.format == "latex":
        latex = analyzer.generate_latex_tables()
        output_path = Path(args.output).with_suffix(".tex")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"LaTeX tables saved to {output_path}")
    else:
        analyzer.save_report(args.output)

    # Print summary to console
    print("\n--- Quick Summary ---")
    findings = analyzer._key_findings()
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    main()
