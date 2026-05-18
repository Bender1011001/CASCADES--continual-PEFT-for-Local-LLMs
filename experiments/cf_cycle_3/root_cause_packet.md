# CF-cycle-3 Root-Cause Evidence Packet — reasoning3 Task 0 batch 93 non-finite loss

## Objective

Diagnose the non-finite training loss that stopped the reduced-memory `reasoning3` control run at Task 0 Epoch 1 Batch 93, without launching treatment or applying a broad training fix.

## Artifacts created

- Static diagnostic script: [`batch93_probe.py`](batch93_probe.py:1)
- Static probe output: [`batch93_static_probe.json`](batch93_static_probe.json:1)

## Key code path

- The guardrail checks `loss = outputs.loss` immediately after model forward and before `loss.backward()` in [`train_cascades()`](../../train.py:347).
- Data preparation masks every prompt token as `-100`, truncates `input_ids` and `labels` independently to `max_length`, and does not require at least one unmasked label token in [`prepare_data()`](../../cascades/data.py:134).
- With `max_length=256`, a sample whose prompt alone is longer than 256 tokens yields a label tensor containing only `-100` values.

## Reproduction-order correction

PyTorch `DataLoader(shuffle=True)` consumes a base iterator seed before the `RandomSampler` seed. A naive `randperm(seed=42)` mapping points to the wrong sample. The static probe in [`batch93_probe.py`](batch93_probe.py:33) reproduces DataLoader order by consuming the base seed first.

Under the corrected order, Task 0 batch 93 is:

- Dataset index: `9` zero-based
- JSONL line: `10` one-based in [`data/task0_gsm8k_cot.jsonl`](../../data/task0_gsm8k_cot.jsonl:10)
- Prompt preview: Rotary Club annual fundraising Omelet Breakfast
- Token stats from [`batch93_static_probe.json`](batch93_static_probe.json:1): `prompt_tokens=304`, `response_tokens=263`, `total_tokens=567`, `label_tokens_after_truncation=0`, `all_labels_masked=true`

## Critical evidence

1. The failed run used `max_length=256` in [`config.json`](../cf_cycle_2/nullspace_ablation/control/config.json:8) and failed at `task=0 epoch=1 batch=93` in [`run_status.json`](../cf_cycle_2/nullspace_ablation/control/run_status.json:13).
2. Batch 93 maps to JSONL line 10, whose prompt alone tokenizes to 304 tokens under the configured tokenizer, exceeding `max_length=256` before any response token can appear in the truncated label span.
3. The probe found exactly three all-ignored-label examples in Task 0 at shuffle positions 93, 132, and 135 in [`batch93_static_probe.json`](batch93_static_probe.json:1). The first such sample is exactly the guardrail failure batch.
4. CPU verification of PyTorch loss semantics shows `torch.nn.functional.cross_entropy(..., ignore_index=-100, reduction='mean')` returns `nan` when every label is ignored. This is independent of model logits, adapter state, optimizer state, or GPU memory.

## Where the non-finite value enters

Confirmed entry point: model forward loss computation for batch 93, before backward, before gradient clipping, before optimizer step, before micro-sleep, and before the next SVC/autopoiesis call at batch 100.

The non-finite value does not require non-finite input tensors or non-finite adapter parameters. A finite logits tensor plus an all-`-100` label tensor is sufficient to produce `NaN` mean cross-entropy.

## Candidate causes reviewed

1. **VRAM/OOM pressure** — contradicted as immediate cause: peak VRAM stayed below threshold in [`run_status.json`](../cf_cycle_2/nullspace_ablation/control/run_status.json:14).
2. **All-labels-masked data batch** — confirmed: batch 93 has zero valid labels after truncation in [`batch93_static_probe.json`](batch93_static_probe.json:1).
3. **Tokenizer/truncation policy** — confirmed contributing mechanism: prompt-first truncation in [`prepare_data()`](../../cascades/data.py:134) can remove all response labels.
4. **SVC/autopoiesis instability** — weak for immediate cause: SVC ran at batch 50 via [`batched_autopoiesis_and_svc()`](../../train.py:388), but the first all-ignored-label batch at 93 is sufficient to generate `NaN` without adapter instability.
5. **Riemannian descent / optimizer state corruption** — weak for immediate cause: failure occurs at the loss guardrail before batch-93 backward or optimizer step in [`train_cascades()`](../../train.py:353).
6. **Quantization/model logits instability** — plausible but not needed for this failure: all-ignored labels produce `NaN` even with zero finite logits.
7. **Invalid tokens or attention mask** — weakened: token IDs are produced by the tokenizer; the decisive invalid condition is zero supervised label tokens, not malformed tokens.

## Distilled diagnosis

**Confirmed root cause:** The reduced `max_length=256` setting exposes a data-loader truncation bug. For Task 0 JSONL line 10, the prompt consumes all 256 retained positions, so every label is `-100`. PyTorch mean cross-entropy over zero valid labels returns `NaN`, and the guardrail correctly aborts at Task 0 Epoch 1 Batch 93.

**Secondary likely cause:** The same issue will recur later in the same epoch at Task 0 batches 132 and 135 if batch 93 is skipped without changing truncation or filtering behavior.

## Cheapest next test

Before any treatment launch, route to Code mode only after diagnosis confirmation and add a minimal CPU test asserting that `prepare_data(..., max_length=256)` never yields an all-`-100` label row for the `reasoning3` Task 0 data order. Then apply the smallest data-loader fix, likely one of:

- Reserve response-label budget by truncating prompt tokens to leave at least one response token.
- Filter/drop examples with zero supervised tokens and persist a count.
- Raise a clear data-preparation error before GPU training when a sample has zero valid labels.

The least ambiguous verification is a CPU-only data-loader scan followed by a bounded control-only GPU repro to batch 100, with per-batch valid-label counts and loss logged. Treatment remains blocked until the control completes finite and under threshold.

## Routing recommendation

- Next route: Code mode for a minimal data-preparation guard/test after human confirmation of this diagnosis.
- Do not route to Experiment Designer for treatment yet.
- Report Writer can summarize this packet after the user confirms the diagnosis or after Code mode adds and verifies the minimal fix.

## Decision status

Treatment remains blocked. Catastrophic forgetting is not solved. This packet explains the failed control run as a data truncation / zero-label-loss failure, not as evidence for or against null-space treatment.
