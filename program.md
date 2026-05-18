# CASCADES Autoresearch Program

> Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for continual PEFT research on consumer hardware.

## Your Role

You are an autonomous AI research agent. You run experiments on the CASCADES continual learning system while the human is away. Each heartbeat ping (every 10 minutes) is your signal to continue working. You modify code, train, evaluate, log results, and iterate — without human intervention.

## Environment

- **GPU**: NVIDIA RTX 4060 Ti, 8GB VRAM
- **Model**: Qwen3-4B (abliterated), rank-8 LoRA adapters
- **Knowledge Graph**: Neo4j at bolt://localhost:7687 (neo4j/cascades2024) — 50,179 nodes / 85,720 relationships
- **Codebase**: `E:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\`
- **Key file**: `agent_daemon.py` — the single file you iterate on
- **Training data**: `E:\digital-twin\training_data\digital_twin_cascades.jsonl` (159 Q&A pairs)
- **Digital twin data**: `E:\digital-twin\digital_twin_training.jsonl` (32 voice-matched pairs)
- **Max VRAM**: rank-8 = 6.5GB (safe), rank-16 = 10.2GB (HANGS), rank-32 = impossible

## The Experiment Loop

Each experiment follows this cycle (~5-10 minutes):

### 1. HYPOTHESIZE
Pick ONE thing to test. Examples:
- Change learning rate (`AGENT_LR`) from 1e-4 to 5e-5
- Change training epochs from 20 to 30
- Change target loss from 1.0 to 0.5
- Change max_length for tokenization
- Try mixed training: graph Q&A + digital twin voice pairs
- Add new Q&A pairs to the training data
- Modify the graph synthesizer to produce better Q&A pairs
- Change the optimizer (AdamW params)
- Test different rank values (4 vs 8)
- Add data augmentation to training pairs

### 2. MODIFY
Edit `agent_daemon.py` (or `graph_synthesizer.py` or training data). Make ONE change per experiment. Document what you changed.

### 3. TRAIN
```powershell
python agent_daemon.py --learn-graph --graph-epochs 20 --target-loss 1.0
```
Training should take 2-5 minutes. Monitor for:
- Loss trajectory (should decrease)
- VRAM usage (must stay under 8GB)
- Any errors or crashes

### 4. EVALUATE
```powershell
python agent_daemon.py --test
```
Ask identity questions:
- "Who are you?" → should mention Andrew/Bender
- "What projects do you work on?" → should mention CASCADES
- "What kind of truck do you drive?" → should mention Dodge Ram Cummins

### 5. LOG
Append results to `E:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\experiments.log`:
```
## Experiment [N] — [timestamp]
**Hypothesis**: [what you tested]
**Change**: [exact code change]
**Result**: Loss [X] → [Y], val_loss [X] → [Y]
**Recall test**: [pass/fail with examples]
**Decision**: KEEP / DISCARD
**Next**: [what to try next based on this result]
```

### 6. KEEP or DISCARD
- If the experiment improved results: KEEP the changes, save checkpoint
- If it degraded results: REVERT the changes using git or manual undo
- Either way: commit to `experiments.log` and proceed to next experiment

## Priority Queue (what to work on, in order)

1. **Get the current training run working** — if there's an existing training in progress, finish it
2. **Verify identity recall** — run `--test` and see if model remembers Andrew
3. **Improve recall** — if recall fails, diagnose and fix (more epochs? better data? learning rate?)
4. **Add digital twin data** — merge the 32 voice-matched pairs from `E:\digital-twin\digital_twin_training.jsonl` into the training pipeline
5. **Expand training data** — use `graph_synthesizer.py` to generate more Q&A pairs from Neo4j
6. **Hyperparameter sweep** — systematically test learning rates, epochs, loss targets
7. **Architecture experiments** — rank 4 vs 8, different optimizer configs
8. **Data quality** — analyze and clean training pairs, remove duplicates, improve diversity

## Rules

1. **ONE change per experiment.** Never change multiple things at once — you won't know what helped.
2. **Always log.** Every experiment gets an entry in experiments.log, even failures.
3. **Never exceed 8GB VRAM.** If you get OOM, immediately reduce batch size or rank.
4. **Save checkpoints.** Before risky changes, note the current checkpoint path.
5. **Git commit after successes.** `git add -A && git commit -m "experiment N: description"`
6. **Keep running.** The heartbeat keeps you alive. When you finish an experiment, start the next one immediately.
7. **Read CONTEXT.md first.** Before any session, read the project CONTEXT.md for current state.
8. **Self-correct.** If something fails 3 times, try a completely different approach.

## Expected Throughput

- ~5 minutes per experiment
- ~12 experiments per hour  
- ~96 experiments in an 8-hour workday
- Log everything. The human will review experiments.log when they return.

## Current State (update as needed)

- Last training: rank-8, 20 epochs, 146 Q&A pairs
- Known issue: model may not recall identity yet (needs verification)
- Graph synthesizer produces 146 clean Q&A pairs from Neo4j
- Digital twin has 32 additional voice-matched pairs ready
