# Bundle Manifest

Created: 2026-05-17

Purpose: compact context bundle for an external advanced math/science LLM to analyze CASCADES catastrophic-forgetting failure modes and propose the next treatment-strength redesign.

## Included files

The bundle includes the selected files listed in [`selected_files.json`](selected_files.json). Files are copied under this directory using their original repository-relative paths.

## Inclusion criteria

- Core code needed to reason about CASCADES update equations, adapter behavior, data loading, metrics, and training.
- Recent experiment evidence needed to understand what failed and what was validated.
- Guardrail tests and comparison code needed to understand which results are valid.
- Project memory and reports needed to prevent stale or overstated conclusions.

## Exclusion criteria

- Large binary artifacts, model weights, notebook files, PDFs, and full training logs are excluded.
- Raw task JSONL files are excluded from this compact pack to keep it small; task names and validation evidence are included through manifests and reports.
- The invalid CF-cycle-1 partial current4 artifacts are not included except through the reports that explain why they are diagnostic-only.

## Current scientific state

- The data-loader all-ignored-label bug was fixed and regression-tested.
- The reduced `reasoning3` control/treatment harness is now capable of producing valid evidence under VRAM and non-finite-loss guardrails.
- Frozen null-space projection activates, has non-empty frozen basis evidence, and removes gradient norm.
- The isolated frozen-nullspace treatment does not meet the BWT success threshold across seeds 42 and 43.
- Next work should redesign treatment strength before escalating to larger suites.

