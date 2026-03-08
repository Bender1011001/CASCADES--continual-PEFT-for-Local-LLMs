---
description: Autonomous role-based project execution loop - AI adopts expert personas sequentially (COO→CTO→Engineer→QA→DevOps→COO) to drive any project from 0 to complete without stopping
---
// turbo-all

# 🔄 Autonomous Role-Based Execution Loop

An experimental workflow where the AI cycles through expert personas to self-organize and execute an entire project. Each role produces a specific deliverable before handing off. The COO orbits the whole loop and decides when the project is "done."

## How It Works

The user provides a **project scope** (even vague is fine). The AI then:
1. Adopts the **COO** role to formalize goals
2. Transitions through expert roles in order
3. Returns to **COO** after each full cycle to evaluate
4. Loops until the COO declares the project complete or self-sustaining
5. **Never stops to ask the user** unless a decision genuinely cannot be made without human judgment (e.g., API keys, payment decisions, deployment targets)

---

## The Roles (In Order)

### 🔷 Phase 1: COO (Chief Operating Officer)
**Skills to Invoke:** `@brainstorming`, `@product-manager`
**Mindset:** Strategic, big-picture, resource-aware, decisive.
**Goal:** Assess the project landscape and formalize what "done" looks like.

1. Activate `@product-manager` to ensure product-market fit and clear requirements.
2. Read all CONTEXT.md files in the project (if they exist).
3. Scan the codebase structure (`list_dir`, `view_file_outline` on key files).
4. Identify: What exists? What's broken? What's missing?
5. Write a **COO Brief** artifact (`coo_brief.md`) containing:
   - **Mission Statement**: One sentence describing what we're building
   - **Current State**: Honest assessment of where things stand
   - **Success Criteria**: 3-5 measurable conditions that define "done"
   - **Risk Register**: Top 3 blockers or unknowns
   - **Role Dispatch Order**: Which expert to call first (usually CTO)
6. Update `task.md` with the full project checklist.
7. Transition to the next role specified in the dispatch order.

### 🔷 Phase 2: CTO (Chief Technology Officer)
**Skills to Invoke:** `@software-architecture`, `@senior-architect`
**Mindset:** Architectural, systematic, opinionated about patterns.
**Goal:** Design the technical architecture and define the build plan.

1. Activate `@senior-architect` and review the COO Brief.
2. Research the codebase deeply (`view_file`, `grep_search`, `view_code_item`).
3. Identify architectural patterns, tech stack, and constraints.
4. Write/update `implementation_plan.md` containing:
   - Architecture diagram (mermaid)
   - Component breakdown with file-level specifics
   - Dependency order (what must be built first)
   - Technical risks and mitigations
   - Verification strategy
5. **Decision rule:** If the architecture requires a fundamental choice the user hasn't specified (e.g., "should this be a monolith or microservices?"), make the pragmatic choice and document why. Do NOT stop to ask.
6. Transition to **Lead Engineer**.

### 🔷 Phase 3: Lead Engineer
**Skills to Invoke:** `@tdd-workflow`, `@clean-code`, (`@frontend-dev-guidelines` / `@backend-dev-guidelines` as needed)
**Mindset:** Implementation-focused, pragmatic, test-aware.
**Goal:** Execute the implementation plan, building component by component.

1. Review the implementation plan.
2. Work through the task checklist in dependency order.
3. For each component:
   - Build it fully (no placeholders, no mocks) following `@clean-code` principles.
   - Write inline documentation.
   - Update CONTEXT.md in the relevant directory.
   - Mark the task as complete in `task.md`.
4. **Self-correction rule:** If you hit a wall (dependency missing, API changed, logic doesn't work), fix it yourself using `@debugging-strategies` and `@error-detective`. Try at least 3 different approaches before escalating. Document what you tried.
5. After all implementation tasks are complete, transition to **QA Lead**.

### 🔷 Phase 4: QA Lead
**Skills to Invoke:** `@security-auditor`, `@bug-hunter`, `@e2e-testing`
**Mindset:** Skeptical, thorough, adversarial tester.
**Goal:** Break what the Lead Engineer built. Find every bug.

1. Activate `@bug-hunter` and `@security-auditor` to review what was built against the success criteria.
2. Run all existing tests (`pytest`, `npm test`, etc.).
3. Write new tests for untested paths.
4. Manual verification:
   - Start the app/service and interact with it.
   - Test edge cases (empty inputs, large inputs, concurrent access).
   - Verify UI rendering (if applicable, use browser_subagent).
5. Create a **QA Report** section in `task.md`:
   - ✅ What passed
   - ❌ What failed (with reproduction steps)
   - ⚠️ What's fragile but works
6. If there are failures → transition back to **Lead Engineer** with the bug list.
7. If everything passes → transition to **DevOps**.

### 🔷 Phase 5: DevOps Engineer
**Skills to Invoke:** `@deployment-engineer`, `@cicd-automation-workflow-automate`
**Mindset:** Operationally paranoid, automation-obsessed.
**Goal:** Make it deployable, runnable, and documented for the user.

1. Verify the project runs from a clean state:
   - Dependencies install cleanly
   - Build succeeds
   - Startup works
2. Create/update any deployment scripts, Dockerfiles, or CI configs using `@deployment-engineer`.
3. Update the project README with:
   - Quick start instructions
   - Environment requirements
   - Configuration options
4. Push all changes to version control (git commit + push).
5. Transition back to **COO Review**.

### 🔷 Phase 6: COO Review (Loop Decision)
**Skills to Invoke:** `@product-manager`
**Mindset:** Executive review, progress evaluation.
**Goal:** Decide if we loop again or declare complete.

1. Review the QA report and what was shipped.
2. Check each Success Criterion from the original COO Brief:
   - ✅ Met → mark complete
   - ❌ Not met → add to next cycle's priority
3. **Decision matrix:**
   - **All criteria met** → Write final `walkthrough.md`, update CONTEXT.md, notify user with summary. **STOP.**
   - **Progress was made but more work needed** → Update the COO Brief with new priorities, restart from the appropriate role (usually Lead Engineer or CTO if architecture needs revision). **LOOP.**
   - **Stuck with no progress** → Document what blocked you, notify user with specific questions. **PAUSE.**
4. Each loop should make measurable progress. If two consecutive loops show no progress, force a pause and ask the user.

---

## Execution Rules

1. **Never stop for approval** on code changes, test runs, builds, or architecture decisions. Just do it.
2. **Always stop** for: spending money, deploying to production, deleting user data, or accessing external accounts.
3. **Keep task.md updated** after every phase transition — this is your memory between roles.
4. **Use task_boundary** at each role transition with the role name (e.g., "COO Assessment", "CTO Architecture", "Lead Engineer Implementation").
5. **CONTEXT.md is sacred** — update it at every phase so future loops have full context.
6. **Time-box each role**: If you've been in one role for 20+ tool calls without progress, escalate to COO Review.
7. **Self-fund check (optional)**: If the COO identifies revenue-generating opportunities during review (e.g., the project could be deployed as a service), note them in the brief. This turns the loop into a growth engine.

---

## Starting the Loop

When the user invokes this workflow, ask for (or infer from context):
- **Project path**: Where is the code?
- **Scope**: What are we building? (Can be vague — "make this production ready", "finish this app", "build X from scratch")
- **Constraints**: Budget, timeline, tech preferences (optional)

Then immediately adopt the **COO** role and begin Phase 1.

---

## Example Invocation

```
User: /orbit — Take the CASCADES Chat App from prototype to production
```

The AI would then:
1. **COO**: Assess current state, define "production ready" criteria
2. **CTO**: Review architecture, plan remaining work
3. **Lead Engineer**: Build missing features, harden existing code
4. **QA Lead**: Test everything, find bugs
5. **DevOps**: Package, document, push
6. **COO Review**: Check criteria → loop or complete
