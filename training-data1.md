# Designing a Think–Plan–Act Curriculum for Sequential Domain Adaptation

## Sequential domain adaptation as a continual learning problem

“Sequential” or “continual” domain adaptation is best framed as a **continual learning** setting where training data arrives in **ordered slices (domains)** and the model must (a) acquire competence on the current slice and (b) retain competence on earlier slices. A representative formalization appears in *continual domain adaptation* benchmarks where domains \(D_1, \dots, D_T\) are presented sequentially and performance is evaluated across all seen domains, not only the latest. citeturn1search12turn1search20turn1search5

A key difficulty is the **stability–plasticity tension**: gradient updates that improve performance on the new domain can interfere with weights supporting earlier domains, producing **catastrophic forgetting**. citeturn1search1turn2search2turn2search10 This is why many sequential/continual training methods introduce explicit retention mechanisms (regularization, replay, parameter isolation, or merge-based schemes). For example, **Elastic Weight Consolidation (EWC)** constrains parameter drift on weights deemed important for prior tasks. citeturn2search2turn2search10

For language models specifically, “domain adaptation” is often implemented via **continued pretraining / domain-adaptive pretraining (DAPT)**, i.e., additional self-supervised pretraining on in-domain corpora after broad pretraining. Empirically, multi-phase adaptive pretraining (DAPT, sometimes followed by task-adaptive pretraining) improves downstream task performance across multiple domains. citeturn2search0turn2search8 When the adaptation is applied repeatedly across a sequence of domains (rather than one target domain), the same forgetting challenges re-emerge, motivating continual-learning-oriented training designs. citeturn2search17turn1search5turn1search20

Your proposed three-stage curriculum (“Think → Plan → Act”) aligns closely with established **reasoning-and-acting** paradigms—most prominently **ReAct**, which interleaves reasoning traces with actions and shows improved performance and interpretability in multi-step tasks that require tool use or environmental interaction. citeturn0search1turn0search14turn0search10 Likewise, chain-of-thought style demonstrations are known to improve performance on multi-step reasoning tasks (especially arithmetic and symbolic reasoning). citeturn0search2turn0search15turn0search11

## Curriculum design principles for a Think–Plan–Act pipeline

A curriculum is not merely “easy to hard”; it is an **ordering policy** that shapes optimization trajectories. The original curriculum learning argument is that gradually increasing difficulty can improve convergence speed and (in non-convex objectives) the quality of solutions found. citeturn1search3turn1search23 In sequential domain adaptation, curriculum design can also be used as a **retention strategy**: early phases establish stable invariants (format adherence, reasoning scaffolds), while later phases introduce more complex behaviors (explicit planning, then tool-executable action) without unlearning earlier invariants.

Operationally, a Think–Plan–Act curriculum can be treated as **three distinct but related domains** presented in sequence:

- **Think domain**: learn a strict boundary marker and a consistent “reasoning summary” style that reliably precedes the final answer. This builds *format discipline* and *verification habits*. (This corresponds to your “Logical Foundation” task.)
- **Plan domain**: learn to embed an explicit decomposition (a numbered plan) inside the thought boundary. This builds *goal-to-subgoal translation* and *risk accounting*. (Your “Architect” task.)
- **Act domain**: learn to emit runnable syntax immediately after the reasoning+plan phase, preventing “reasoning regularization” from suppressing coding/tool execution fluency. (Your “Worker” task; tightly related to ReAct-style trajectories). citeturn0search1turn0search14turn0search10

Two design tensions matter for sequential domain adaptation:

**Interference risk across stages.** If later stages reward verbosity or introduce conflicting formatting norms, they can overwrite earlier formatting constraints—classic catastrophic forgetting dynamics. citeturn1search1turn2search2turn2search17 Your “never deviate from schema” objective therefore needs *hard constraints* (validators) and *anti-drift anchors* (held-out schema tests and periodic re-exposure to earlier-stage formats).

**Ambiguity control.** Planning and acting tasks are often under-specified: multiple valid decompositions, multiple feasible architectures, multiple shell command variants. Without careful rubric design, training rewards can become inconsistent, which increases mode collapse or brittle adherence (“the model outputs a plan because that’s rewarded, but the plan doesn’t actually align with requirements”). ReAct-style work highlights that coupling reasoning with environment interactions can reduce hallucination and error propagation—but only if the action space and success criteria are well-defined. citeturn0search1turn0search14turn0search10

## Schema and formatting requirements for robust JSONL training

Many supervised fine-tuning pipelines consume **JSON Lines (JSONL)** where each line is a standalone JSON object representing one training example. citeturn2search28turn0search28turn0search24 Practical SFT ecosystems commonly accept “prompt/response” pairs or instruction-style variants, but the underlying constraint is the same: each line must be valid JSON and independently parseable. citeturn0search20turn0search24turn0search28

Because your objective is *schema-cementing*, treat formatting as an explicit part of the learning target:

**Canonical JSON object per line.** A minimal, interoperable representation is:
- `prompt`: string
- `response`: string

This matches a broadly used instruct-style pair format discussed in SFT format guides. citeturn0search24turn0search28turn0search20

**Escaped newlines.** When you say “linebreaks mapped to `\n`”, you are imposing a serialization constraint: the JSON string literal contains `\\n` characters (escape sequences) rather than raw newlines. That is a reasonable constraint for strict parsers and makes line-based processing safer, but it must be enforced by generation-time escaping and validated by a linter.

**Important spec inconsistencies to resolve before generation.** Your request contains three direct conflicts that will break “never deviate from schema” if not resolved:

- You specify “Each dataset must contain **50**” examples, but later request **100** for the second and third datasets.
- You request datasets “clearly separated by **comments or markdown headers**,” yet also require “Do not output anything outside the JSONL blocks.” JSONL does not support comments; header lines would violate strict JSON-per-line parsing.
- You require “Task 0 has no `<plan>` or `<action>`”, but the overall “Cognitive Architecture” says every turn has an embedded `<plan>` inside `<thought>`. These are mutually exclusive definitions of the target behavior.

From a dataset engineering standpoint, these should be resolved by treating “Task 0 / Task 1 / Task 2” as **separate files** (e.g., `task0.jsonl`, `task1.jsonl`, `task2.jsonl`) with internally consistent rules per file—precisely because JSONL is inherently file-scoped, not multi-dataset-scoped. citeturn0search28turn0search24turn0search20

**Schema validators as part of the training pipeline.** For your “airtight cognitive pipeline,” you want *hard gating* before a sample enters the training set:
- JSON parse check
- Must contain only keys `prompt`, `response`
- `response` must match an exact tag order per task (regex + structured parse)
- Newline escaping rule check
- For Task 2, code fence presence check and (optionally) a sandboxed syntax check (e.g., `bash -n`, `python -m py_compile`)—action-heavy datasets are extremely sensitive to small formatting drift.

These mechanics are standard in robust SFT data preparation pipelines: you clean/transform, then validate, then export final JSONL. citeturn0search12turn0search24turn0search28

## Logical foundation dataset design

Your “Logical Foundation” dataset is effectively a **format-conditioning + deductive-reasoning** domain. Chain-of-thought demonstrations are known to improve multi-step reasoning accuracy, particularly for arithmetic and symbolic tasks. citeturn0search2turn0search11turn0search15 However, for production systems and many training regimes, it is often preferable to distinguish between (a) a concise, user-facing explanation and (b) a private scratchpad, because long free-form rationales can encourage leakage of internal reasoning traces and can become brittle across domains.

If your goal is strict schema adherence without encouraging uncontrolled verbosity, a high-precision approach is to define `<thought>` content as a **structured reasoning summary** with explicit intermediate values, plus a short verification line, rather than unconstrained narrative. This preserves the “think-first” behavior while reducing drift.

A rigorous Task-0 prompt distribution that supports sequential domain adaptation should include:
- **Multi-constraint word problems** (ratios, mixtures, piecewise pricing, time-rate-distance with schedule constraints)
- **Paradox-style logic puzzles** (liar/truthteller with multiple agents, self-reference, constrained truth tables)
- **Algorithmic complexity reasoning** (Big-O with nested loops, amortized complexity, worst/best/average with clear assumptions)
- **Formal deduction** (syllogisms, implication chains, contradiction proofs)

To prevent the model from learning superficial patterns, incorporate:
- Redundant numerical details (forcing selection of relevant quantities)
- Counterfactual checks (“if X were 1 more, how would the total change?”)
- Explicit unit-checking and bounds checking

This aligns with the underlying intent of chain-of-thought prompting—explicit intermediate reasoning steps—shown to yield strong gains on arithmetic tasks like GSM-style word problems. citeturn0search2turn0search15

## Architect dataset design

Your “Architect” dataset targets the ability to (a) reason about ambiguous engineering goals and (b) produce a deterministic decomposition. This is essentially teaching **planning under uncertainty** and **risk-aware execution structure**, which is central to modern agentic workflows. citeturn0search6turn0search1turn0search14

The core difficulty is that architecture/planning problems rarely have a single ground truth. To make a dataset “rigorous” under ambiguity, you need to build **rubrics into the prompt** so that a “best” plan is identifiable. Examples of rubric hooks:
- explicit SLOs (latency, availability, RPO/RTO)
- compatibility constraints (language/runtime versions, regulatory requirements)
- rollout constraints (zero downtime, blue/green, canary)
- data constraints (multi-tenant isolation, PII handling, backfill windows)
- observability constraints (metrics/logs/traces, auditability)

Within the `<plan>` you want discrete steps that are:
- **Atomic**: each step has one primary verb and a measurable outcome
- **Order-constrained**: dependencies stated implicitly by ordering
- **Reversible**: rollback paths or safe-guards explicitly included
- **Verifiable**: each step has an acceptance check

This is consistent with the ReAct motivation that reasoning traces help maintain and update plans, while actions interface with tooling; in your Task 1 you omit actions but still want the planning discipline. citeturn0search1turn0search14

## Worker dataset design

Your “Worker” dataset is the **execution domain**: after reasoning and planning, the system must output runnable code. In sequential domain adaptation terms, this stage has high interference risk: the model may over-index on explanation and under-deliver on syntactic precision unless the dataset heavily reinforces exactness.

Three best-practice pillars for action datasets:

**Idempotency and safety.** System administration and automation tasks should prefer idempotent commands where possible (e.g., `mkdir -p`, conditional checks before destructive operations) and must include explicit constraints about environment assumptions. This reduces the likelihood of brittle scripts and aligns with reliable agent behavior. ReAct-style work emphasizes that tool interaction can reduce hallucination—but only if actions are grounded and verifiable. citeturn0search1turn0search14

**OS and shell variance.** Prompts should specify OS (Linux/macOS), distribution, shell (`bash` vs `zsh`), and whether `sudo` is available, because the exact runnable syntax differs materially. This is directly aligned with your requirement “Reflect on the user’s OS and constraints” inside the thought/planning phase.

**Post-action verification hooks.** Even when the dataset ends at producing code, requiring the action block to include validation commands (e.g., `curl -f`, `git status`, `python -m py_compile`, exit-code checks) increases real-world reliability and reduces silent failure.

Finally, if your broader goal is continual adaptation without catastrophic forgetting, you should expect that code-centric skills can degrade if the later “thinking” stage becomes too dominant. Continual learning work repeatedly identifies interference and forgetting as core risks in sequential training. citeturn1search1turn2search17turn2search2 This is exactly why your Task 2 exists—and why it should be weighted and validated independently (syntax checks, execution tests) rather than judged purely by textual similarity.

## Validation and evaluation for sequential training

A “never deviate from schema” objective is best treated as a **measurable contract** rather than an aspiration. A robust evaluation framework should include:

**Schema adherence tests (per task).**
- exact tag ordering checks
- forbidden tag checks (e.g., Task 0 must not contain `<plan>` or `<action>`)
- code fence integrity checks (Task 2)
- escaped-newline compliance checks

**Functional correctness tests.**
- For Task 0: numeric answer verification via deterministic solvers for a subset of problems; complexity reasoning can be unit-tested with canonical loop patterns.
- For Task 2: sandboxed execution tests under constrained inputs; at minimum, syntactic checks (`bash -n`, `python -m py_compile`) and static checks for obviously unsafe constructs.

**Sequential retention metrics.**
Because the core risk is forgetting, evaluate after each stage on:
- the current stage’s held-out set
- *all prior stages’* held-out sets, tracking degradation (average accuracy drop, worst-case drop)

This approach follows the continual learning perspective: models trained sequentially must maintain performance on previously learned distributions and are otherwise deemed to have forgotten. citeturn1search5turn1search12turn2search2

**Mitigation levers when retention drops.**
The literature provides several families of interventions:
- regularization-based retention (e.g., EWC) citeturn2search2turn2search10  
- replay / rehearsal / mixture strategies (e.g., continual pretraining evaluation setups emphasize measuring forgetting under naive fine-tuning baselines) citeturn2search17  
- parameter-efficient isolation approaches (e.g., LoRA-based continual learning methods and low-rank subspace approaches) citeturn2search35turn2search27turn2search39

The practical takeaway for your curriculum is: treat “Think,” “Plan,” and “Act” as **separate but revisitable domains**, and design your sampling/weighting so that earlier-stage format invariants are periodically re-reinforced, while later-stage code exactness is continuously unit-tested. This is the most direct way to align dataset design with the core failure mode of sequential domain adaptation: catastrophic forgetting. citeturn1search1turn2search2turn2search17