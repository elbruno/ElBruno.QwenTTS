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
