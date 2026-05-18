# CASCADES Lifelong Agent — Task List

## Current: Graph → Parametric Memory Training
- [x] Fix 5 critical bugs (save bug, prompt collision, gradient leak, eval mismatch, EOS truncation)
- [x] Clean graph synthesizer (filter NPM noise, unique prompts, grammar fixes)
- [ ] **IN PROGRESS**: Run rank-8, 20-epoch training on 146 clean Q&A pairs (target loss < 1.0)
- [ ] Run identity recall test (`--test`) — verify model remembers Andrew/Bender1011001
- [ ] If recall fails: diagnose, fix, retrain
- [ ] If recall succeeds: proceed to VM deployment

## Next: Autonomous VM Loop with Logging
- [ ] Start agent in interactive mode with VM idle loop enabled
- [ ] Add comprehensive logging to agent_daemon.py:
  - Log every VM command attempted + output to `vm_activity.log`
  - Log every dream cycle (what was learned, loss before/after)
  - Log every subspace freeze (which dimensions, what facts)
  - Timestamp everything
- [ ] Let the model explore and learn autonomously
- [ ] Monitor logs to ensure it's learning, not degrading
- [ ] Periodic identity recall tests to verify no catastrophic forgetting

## Future
- [ ] Clean up Neo4j graph (remove noise entities like "Password", "Sendername")
- [ ] Add more high-value identity facts to graph_synthesizer hardcoded list
- [ ] Consider renting H100 for rank-32 training if rank-8 capacity fills up
- [ ] Implement temporal episode consolidation (daily summaries)

## Notes
- Rank 8 = 6.5GB VRAM, runs fine on 4060 Ti 8GB
- Rank 16 = 10.2GB, hangs during loading (VRAM thrash)
- Rank 32 = way too much, don't even try on this card
- Always use `--test` mode for evaluation (matches training rank)
- Never use `--fresh` if a valid checkpoint exists at the same rank
