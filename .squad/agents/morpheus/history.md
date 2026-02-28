# Project Context

- **Owner:** Bruno Capuano
- **Project:** Qwen3-TTS → ONNX → C# .NET 10 console app for local voice generation
- **Stack:** Python (PyTorch, transformers, ONNX), C# (.NET 10, ONNX Runtime), Qwen3-TTS
- **Created:** 2026-02-21T15:38Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-02-21: GPU Handoff
Project published to https://github.com/elbruno/qwen-labs-cs (private). 8/12 tasks complete. All scaffolding done — both Python scripts and C# code. Remaining work is execution (run exports) and implementation (LM inference in C#). Key risk: export_lm.py hasn't been validated against real model weights yet.

### 2026-02-28: Issue #22 Triage — Security/Performance/CI Audit
**By:** Morpheus
**What:** Assessed Issue #22 (lessons from elbruno.localembeddings v1.1.0) and created 3-phase roadmap: Phase 1 (NOW: Security — 4 items), Phase 2 (NEXT: Performance — 4 items), Phase 3 (LATER: CI — 2 items). Triaged work breakdown in `.squad/decisions/inbox/morpheus-issue-22-triage.md`. Key findings: (1) Path traversal risk is low (hardcoded repIds); (2) Input validation is **blocking** for voice cloning release (SEC-1); (3) Top-K sampling optimization can yield 5–10% latency gain (PERF-1); (4) CI validation for git tags is low-priority (manual tag creation).
**Why:** Security > performance > CI in priority. SEC-1 (text input validation) unblocks VoiceCloning Web feature. Neo owns all Phase 1 security fixes with Morpheus code review. Phase 2/3 deferred to prevent scope creep.

### 2026-02-28: Neo's Queue
📌 **Team update:** Issue #22 triage completed. Created 3-phase roadmap for security/performance/CI audit. Phase 1 (security) is blocking for voice cloning release; Phase 2/3 deferred. SEC-1 (input validation) now in Neo's queue. — Morpheus

### 2026-02-28: Phase 2 & 3 Merge Review — All Branches Complete
**By:** Morpheus (Lead)
**What:** Reviewed and merged all Phase 2 (Performance) and Phase 3 (CI/Linux) branches to main:
  - **PERF-1 (squad/perf-1-topk-heap):** Top-K heap speaker search — O(n log k) optimization with SIMD acceleration. Clean implementation with 21 tests. Min-heap pattern is textbook-correct.
  - **PERF-2 (squad/perf-2-arraypool):** ArrayPool adoption in LanguageModel inference — reduces GC pressure during autoregressive generation. Proper rent/return lifecycle, no leaks.
  - **PERF-3 (squad/perf-3-benchmarks):** BenchmarkDotNet baseline infrastructure — enables continuous performance validation. Good test coverage for benchmark execution.
  - **Phase 3 (squad/phase-3-ci-linux):** CI/Linux workflow hardening — version detection from GitHub releases, manual dispatch, csproj fallback. Linux validation added to build matrix.
**Merge Strategy:** Squash merge with conventional commits (fix(perf), fix(build), fix(ci)) referencing #22. Each commit includes Co-authored-by trailer per team convention.
**Validation:** All 4 branches validated independently (60 tests, 0 warnings, 0 errors). Final merge validation on main confirms no integration issues.
**PR Review Patterns:** Validated build output, test metrics, implementation quality, and architecture decisions for each branch. Code quality high across all work items.
**Why:** Systematic review ensures production readiness. Squash commits maintain clean git history. All work traceable to Issue #22.
**Closure Decision:** Issue #22 closed after all work merged. Phase 1 (SEC-1/2/3/4) + Phase 2 (PERF-1/2/3) + Phase 3 (CI) complete. Total 7 work items resolved.
**Learnings:** Squash merge strategy works well for multi-agent branches. Review gates (build + test validation) catch integration issues early. Decision memo consolidation provides audit trail for all architectural choices.
