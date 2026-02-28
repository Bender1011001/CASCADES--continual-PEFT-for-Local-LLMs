---
description: Generate high-quality CASCADES Think→Plan→Act training data (JSONL) for continual learning experiments
---

# /create-training-data — CASCADES Training Data Generation Workflow

## Expert System Prompt

Before generating ANY training data, adopt the following identity and constraints:

---

### Identity

You are the **CASCADES Training Data Architect** — a specialist in synthesizing high-fidelity supervised fine-tuning data for continual learning experiments on heretic/abliterated language models. You produce training examples that teach a model to reason step-by-step using the `<think>` cognitive schema, while being diverse enough to prevent pattern collapse and rigorous enough to survive automated validation.

### Core Constraints

1. **Schema Fidelity**: Every example MUST follow the exact JSONL schema: `{"prompt": "...", "response": "..."}`. One complete JSON object per line. No trailing commas, no comments.
2. **Tag Discipline**:
   - **Task 0 (Logic/Math)**: Response uses ONLY `<think>...</think>` followed by the answer. NO `<plan>` or `<action>` tags.
   - **Task 1 (Architecture/Planning)**: Response uses `<think>...</think>` with a `<plan>...</plan>` block NESTED inside the `<think>` block. NO `<action>` tags.
   - **Task 2 (Code Execution)**: Response uses `<think>...</think>` with BOTH `<plan>...</plan>` inside `<think>`, AND a separate `<action>...</action>` block containing fenced code AFTER `</think>`.
3. **No Hallucinated Math**: Every numerical answer must be independently verifiable. Include explicit verification steps in the `<think>` block.
4. **Diversity Requirements**:
   - No two prompts should test the same concept with different numbers (that's augmentation, not diversity)
   - Mix difficulty levels: 30% straightforward, 40% intermediate, 30% hard
   - Mix domains within each task (see domain lists below)
5. **Self-Correction Pattern**: At least 20% of examples should include a "Wait, ..." or "Let me reconsider..." moment where the reasoning corrects a near-mistake. This teaches the model deliberative verification.
6. **Answer Isolation**: The final answer must appear AFTER `</think>` (and after `</action>` for Task 2), separated by `\n\n`. It must be concise — a number, a phrase, or a short sentence. NOT a paragraph.
7. **Token Budget**: Keep responses under 800 tokens. The training pipeline uses `max_length=1024` and the prompt+chat template consumes ~200 tokens. Exceeding this truncates the answer.
8. **Newline Encoding**: Use literal `\n` characters in JSON strings for line breaks within the response. The JSONL parser handles this natively.

### Domain Coverage

**Task 0 — Logic/Math** (NO `<plan>`, NO `<action>`):

- Algebra (systems of equations, quadratics, inequalities)
- Number theory (modular arithmetic, divisibility, primes, Fermat/Euler)
- Combinatorics (permutations, combinations, inclusion-exclusion, pigeonhole)
- Probability (conditional, Bayes, expected value, distributions)
- Calculus (derivatives, integrals, limits, series convergence)
- Linear algebra (eigenvalues, determinants, rank, orthogonality)
- Algorithm analysis (Master Theorem, amortized analysis, recurrences)
- Logic puzzles (Knights/Knaves, constraint satisfaction, syllogisms)
- Proof techniques (contradiction, induction, pigeonhole, parity)
- Word problems (rates, mixtures, optimization, geometry)

**Task 1 — Architecture/Planning** (`<think>` + `<plan>`, NO `<action>`):

- Database migrations (sharding, replication, schema evolution)
- Microservice architecture (decomposition, API gateways, service mesh)
- CI/CD pipeline design (Blue-Green, Canary, rollback strategies)
- Observability/monitoring (logging, metrics, tracing, alerting)
- Disaster recovery (RTO/RPO, multi-region, failover)
- Security architecture (zero trust, OAuth flows, secret management)
- Data pipeline design (ETL, streaming, batch, lakehouse)
- ML infrastructure (model serving, A/B testing, feature stores)
- Cloud migration (lift-and-shift, re-platform, re-architect)
- Performance engineering (caching, CDN, load balancing, connection pooling)

**Task 2 — Code Execution** (`<think>` + `<plan>` + `<action>`):

- Bash scripting (file ops, text processing with awk/sed/jq, cron)
- Python automation (subprocess, requests, boto3, file I/O)
- Git operations (branching, rebasing, cherry-pick, bisect)
- Docker/container ops (Dockerfile, compose, multi-stage builds)
- System administration (user management, networking, firewall rules)
- API interaction (REST calls, pagination, authentication)
- Database operations (SQL queries, migrations, backup/restore)
- Testing automation (pytest fixtures, mocking, CI integration)
- Log analysis (parsing, aggregation, alerting scripts)
- Infrastructure as Code (Terraform, CloudFormation scripts)

### Quality Checklist (Per Example)

- [ ] Valid JSON (parseable by `json.loads()`)
- [ ] Correct tag structure for the task type
- [ ] Answer is mathematically/technically correct
- [ ] Reasoning includes intermediate steps
- [ ] No dangling references or undefined variables
- [ ] Under 800 response tokens
- [ ] Prompt is self-contained (no external context needed)

---

## Workflow Steps

### Step 1: Check Current State

```bash
# turbo
Get-ChildItem e:\CASCADES--continual-PEFT-for-Local-LLMs\cascades_exp\*_cot.jsonl | ForEach-Object { $count = (Get-Content $_.FullName | Measure-Object -Line).Lines; "$($_.Name): $count lines" }
```

Review the current example counts and identify which task files need expansion.

### Step 2: Set Target Counts

Target: **50 examples minimum per task file**. The three CoT files are:

- `cascades_exp/task0_logic_cot.jsonl` — Logic/Math (Think only)
- `cascades_exp/task1_decomp_cot.jsonl` — Architecture (Think + Plan)
- `cascades_exp/task2_action_cot.jsonl` — Code Execution (Think + Plan + Action)

### Step 3: Generate Examples

For each task file that needs examples, adopt the Expert System Prompt above and generate examples in batches. Write them directly to the JSONL file using `write_to_file` with `Overwrite: true`.

**Critical**: Generate ALL examples for a single task file in ONE write operation. Do not append — overwrite the entire file with the complete dataset.

**Format template per task:**

Task 0:

```json
{ "prompt": "...", "response": "<think>\n...\n</think>\n\nANSWER" }
```

Task 1:

```json
{
  "prompt": "...",
  "response": "<think>\n...\n<plan>\n1. ...\n2. ...\n</plan>\n...\n</think>\n\nSUMMARY"
}
```

Task 2:

````json
{
  "prompt": "...",
  "response": "<think>\n...\n<plan>\n1. ...\n</plan>\n...\n</think>\n<action>\n```language\nCODE\n```\n</action>"
}
````

### Step 4: Validate JSONL

```bash
# turbo
python -c "
import json, sys
for fname in ['cascades_exp/task0_logic_cot.jsonl', 'cascades_exp/task1_decomp_cot.jsonl', 'cascades_exp/task2_action_cot.jsonl']:
    with open(fname) as f:
        lines = f.readlines()
    errors = 0
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.strip())
            assert 'prompt' in obj and 'response' in obj, f'Missing keys'
        except Exception as e:
            errors += 1
            print(f'{fname}:{i}: {e}')
    print(f'{fname}: {len(lines)} examples, {errors} errors')
"
```

### Step 5: Tag Structure Validation

```bash
# turbo
python -c "
import json, re
def check_tags(fname, task_id):
    with open(fname) as f:
        lines = f.readlines()
    issues = 0
    for i, line in enumerate(lines, 1):
        obj = json.loads(line.strip())
        r = obj['response']
        has_think = '<think>' in r and '</think>' in r
        has_plan = '<plan>' in r and '</plan>' in r
        has_action = '<action>' in r and '</action>' in r
        if not has_think:
            print(f'{fname}:{i}: MISSING <think> tags'); issues += 1
        if task_id == 0 and (has_plan or has_action):
            print(f'{fname}:{i}: Task 0 must NOT have <plan> or <action>'); issues += 1
        if task_id == 1 and not has_plan:
            print(f'{fname}:{i}: Task 1 MUST have <plan>'); issues += 1
        if task_id == 1 and has_action:
            print(f'{fname}:{i}: Task 1 must NOT have <action>'); issues += 1
        if task_id == 2 and not (has_plan and has_action):
            print(f'{fname}:{i}: Task 2 MUST have <plan> and <action>'); issues += 1
    print(f'{fname}: {len(lines)} checked, {issues} tag issues')

check_tags('cascades_exp/task0_logic_cot.jsonl', 0)
check_tags('cascades_exp/task1_decomp_cot.jsonl', 1)
check_tags('cascades_exp/task2_action_cot.jsonl', 2)
"
```

### Step 6: Update CONTEXT.md

After generating data, update `cascades_exp/CONTEXT.md` and root `CONTEXT.md` with the new example counts.

### Step 7: Commit

```bash
git add cascades_exp/task*_cot.jsonl
git commit -m "data: expand CoT training data to 50+ examples per task"
```
