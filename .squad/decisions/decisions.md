# Team Decisions

All team decisions are recorded here. Decisions are merged from `.squad/decisions/inbox/` by Scribe and deduplicated.

---

### 2026-02-28: Issue #22 Triage — Security, Performance & CI Audit

**Issue:** Apply security, performance & CI lessons from LocalEmbeddings v1.1.0 audit  
**Date:** 2026-02-28  
**By:** Morpheus  

#### What

Assessed Issue #22 and created 3-phase roadmap:

- **Phase 1 (NOW): Security** — 4 items (SEC-1 through SEC-4)
  - SEC-1: Input validation on TtsPipeline.SynthesizeAsync (text length, encoding, null checks)
  - SEC-2: Path traversal validation in VoiceCloningDownloader (reject `..` segments)
  - SEC-3: ONNX/NPY file size pre-checks (enforce size limits)
  - SEC-4: Document HTTPS enforcement for HuggingFace downloads

- **Phase 2 (NEXT SPRINT): Performance** — 4 items (PERF-1 through PERF-4)
  - PERF-1: Optimize SampleToken with top-K heap (O(n log n) → O(n log k), 5–10% latency gain)
  - PERF-2: ArrayPool for temporary embeddings in LanguageModel
  - PERF-3: Add BenchmarkDotNet benchmarks (latency/allocation)
  - PERF-4: TensorPrimitives for softmax (optional, research-backed)

- **Phase 3 (LATER): CI/Linux** — 2 items (CI-1 through CI-2)
  - CI-1: Validate git tag format in publish.yml
  - CI-2: Add Windows CI to publish workflow

#### Why

- **Security > performance > CI** in priority
- SEC-1 (input validation) **blocks VoiceCloning release**
- Path traversal risk is LOW (hardcoded `repoId`); input validation is HIGH-impact
- Performance optimizations don't block features; defer to Phase 2
- CI validation is low-priority; manual tag creation by Bruno is low-risk

#### Key Decisions

1. No scope for Model Integrity (SHA-256 hashing) — deferred to v1.1.0
2. Input validation is **blocking** for voice cloning feature
3. Performance optimization requires benchmarks (no guesses)
4. **Neo owns Phase 1 security fixes** (implementation)
5. **Morpheus owns Phase 1 code review** of Neo's PRs
6. **Tank validates Phase 2 benchmarks** (BenchmarkDotNet required)
7. Phase 2/3 deferred to prevent scope creep

#### Related Files

- Full triage report: `.squad/decisions/inbox/morpheus-issue-22-triage.md`
- Morpheus history: `.squad/agents/morpheus/history.md`
