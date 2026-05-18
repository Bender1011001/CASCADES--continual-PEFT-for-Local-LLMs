---
description: Autonomous experiment loop inspired by karpathy/autoresearch. Modifies code, trains, evaluates, keeps or discards, and repeats -- running all day without human input.
---

# Autoresearch Workflow

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The core idea: give the agent a training setup and let it experiment autonomously. It modifies code, trains for a fixed time budget, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) a better model.

## Prerequisites

1. A `program.md` in the project root describing:
   - What file(s) the agent can modify
   - How to train (the command)
   - How to evaluate (the command + what metric to check)
   - Hardware constraints (VRAM, time budget)
   - Priority queue of experiments to try
2. An `experiments.log` file (created automatically if missing)
3. The heartbeat running (`/heartbeat` workflow)

## Setup

// turbo
1. Start the heartbeat if not already running:
```
Start-Process cmd -ArgumentList '/c','title Antigravity Heartbeat && node E:\code.projects\antigravity-automation\heartbeat.js --interval 10' -WindowStyle Normal
```

2. Create or verify `program.md` exists in the project root. It should contain:
   - **Environment**: GPU, model, key files, constraints
   - **The Experiment Loop**: hypothesize → modify → train → evaluate → log → keep/discard
   - **Priority Queue**: ordered list of experiments to try
   - **Rules**: one change per experiment, always log, save checkpoints

3. Update `E:\code.projects\antigravity-automation\tasks.md` to point at this project:
```markdown
# Current Directive: Autoresearch

Follow [PROJECT_DIR]/program.md for experiment instructions.

### On each heartbeat:
1. Read program.md for instructions
2. Read experiments.log for what's been done
3. Read CONTEXT.md for current state
4. Run the next experiment in the loop
5. Log results to experiments.log
```

## The Experiment Loop (what the agent does on each heartbeat)

### Step 1: Read State
- Read `program.md` for instructions
- Read `experiments.log` to see which experiments have been run
- Read `CONTEXT.md` for current project state
- Identify the next experiment from the priority queue

### Step 2: Hypothesize
- Pick ONE thing to test based on the priority queue and previous results
- Write the hypothesis before making any changes

### Step 3: Modify
- Make ONE change to the target file(s)
- Never change multiple things at once — you won't know what helped

### Step 4: Train
- Run the training command from program.md
- Monitor for errors, OOM, crashes
- If training fails, log the failure and try a different approach

### Step 5: Evaluate
- Run the evaluation command from program.md
- Compare the metric against the baseline and previous best
- Record exact metric values

### Step 6: Log
Append to `experiments.log`:
```markdown
## Experiment N — YYYY-MM-DD HH:MM
**Hypothesis**: [what you're testing]
**Change**: [exact code/config change, ideally a diff]
**Training**: [loss trajectory, time taken, any issues]
**Evaluation**: [metric before → after]
**Decision**: KEEP / DISCARD
**Reasoning**: [why this result makes sense]
**Next**: [what to try next based on this result]
```

### Step 7: Keep or Discard
- **KEEP**: Leave the changes, save/commit checkpoint
- **DISCARD**: Revert changes (git checkout or manual undo)

### Step 8: Repeat
- Immediately start Step 1 again for the next experiment
- Target: ~12 experiments per hour (5 min each)
- The heartbeat ensures you keep running

## Key Principles (from karpathy/autoresearch)

1. **Single file to modify.** Keep the scope small. One file (or a small set) that the agent iterates on.
2. **Fixed time budget.** Each experiment takes the same amount of wall clock time, making results comparable.
3. **Self-contained.** No external dependencies beyond what's already installed. One GPU, one file, one metric.
4. **One change per experiment.** Scientific method — control your variables.
5. **Always log.** Every experiment gets recorded, even failures. Failures are data.
6. **Self-correct.** If something fails 3 times, pivot to a completely different approach.

## Example program.md Template

```markdown
# [Project] Autoresearch Program

## Environment
- GPU: [model, VRAM]
- Model: [what you're training]
- Key file: [the file the agent edits]
- Training data: [path]
- Constraints: [VRAM limits, time budget]

## Train Command
\`\`\`
[exact command to run training]
\`\`\`

## Eval Command
\`\`\`
[exact command to evaluate]
\`\`\`

## Metric
[what number to optimize, lower/higher = better]

## Priority Queue
1. [First thing to try]
2. [Second thing to try]
3. [Third thing to try]

## Rules
- One change per experiment
- Always log to experiments.log
- Never exceed [VRAM] GB
- Save checkpoints before risky changes
```

## Stopping

The loop runs until:
- The heartbeat is stopped (`/heartbeat` stop step)
- The human sends a message
- All items in the priority queue are exhausted (add more!)

## Tips

- **Start simple.** First experiments should verify the baseline works at all.
- **Learn from failures.** A failed experiment that shows X doesn't work is valuable.
- **Build on successes.** If lowering LR helped, try lowering it more.
- **Watch for overfitting.** If training loss improves but eval gets worse, you're overfitting.
- **Commit good checkpoints.** Use git or copy checkpoint files before risky experiments.
