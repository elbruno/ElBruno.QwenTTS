# Issue #22 Triage: Security, Performance & CI Audit

**Issue:** Apply security, performance & CI lessons from LocalEmbeddings v1.1.0 audit  
**Date:** 2026-02-28  
**Morpheus Decision:** See below  

---

## Executive Summary

Issue #22 presents a comprehensive security/performance/CI audit checklist adapted from elbruno/elbruno.localembeddings#38. **I assessed that this repo will benefit from targeted security work immediately, performance work in phase 2, and CI/Linux work in phase 3.** We have 5 active projects (Core, Web, VoiceCloning, CLI, FileReader); prioritization focuses on Core (NuGet package) and public-facing APIs first.

---

## Assessment by Category

### 🔒 Security Checklist

**Applies to this repo: YES (partially)**

Current state:
- ✅ **URL validation**: VoiceCloningDownloader uses hardcoded HTTPS base (`https://huggingface.co`); Core delegates to ElBruno.HuggingFace library (trusted dependency).
- ⚠️ **Path traversal**: ModelDownloader and VoiceCloningDownloader construct paths via `Path.Combine(modelDir, fileName)` with untrusted `repoId`. Risk is **low** because:
  - `repoId` is hardcoded defaults or set by developer, not user input
  - VoiceClonePipeline accepts `referenceAudioPath` from user (WAV files) — **no validation**
  - FileReader reads user-supplied file paths for batch processing — **no validation**
- ❌ **File validation** (size/format): ONNX models are loaded directly without size checks. NPY files are parsed without bounds checking (NpyReader).
- ❌ **Input validation** on public APIs: TextTokenizer, LanguageModel, and Vocoder accept user text without length limits or encoding checks.
- ❌ **Cross-platform file name validation**: Not used yet, but future APIs may need it.

**In Scope (Phase 1 — Security):**
1. **Add input validation to TTS pipeline** (text length, encoding, null checks) — critical for Web/CLI/VoiceCloning
2. **Add path traversal checks** in VoiceCloningDownloader (reject `..` segments, enforce relative paths)
3. **Add file size pre-checks** for ONNX/NPY files before loading
4. **Document HTTPS enforcement** for all HuggingFace downloads

**Out of Scope (deferred):**
- Cross-platform file name validation (not needed yet; can defer to next audit)
- Deep model integrity checks (SHA-256 hashing of downloaded files — scope creep for v1.0)

---

### ⚡ Performance Checklist

**Applies to this repo: PARTIALLY**

Current state:
- ❌ **TensorPrimitives**: Not used. LanguageModel uses manual loops for vector operations (normalization, softmax, top-K sampling). **High opportunity** for SIMD speedup.
- ⚠️ **Span / Memory**: Good use of Buffer.BlockCopy and manual arrays in hot paths; but many allocation opportunities in LanguageModel decode loop.
- ❌ **ArrayPool**: Not used. Temporary buffers (flatEmbeds, cpInputs, cpEmbedBuf) are allocated fresh each iteration.
- ❌ **Top-K search**: LanguageModel.SampleToken() uses LINQ OrderByDescending().Take(topK) — inefficient O(n log n) instead of O(n log k).
- ❌ **Benchmarks**: None exist. No BenchmarkDotNet suite.

**In Scope (Phase 2 — Performance, deferred to next sprint):**
1. **Optimize SampleToken with top-K heap** (replace LINQ sort with manual min-heap) — 5–10% latency gain
2. **Pool temporary buffers in LanguageModel** (ArrayPool for embed buffers in decode loop)
3. **Add BenchmarkDotNet benchmarks** for inference latency and allocations
4. **TensorPrimitives for softmax/normalization** (lower priority; need profiling)

**Out of Scope (nice-to-have):**
- Full rewrite with MathNet.Numerics or TensorFlow.NET (scope creep)
- GPU inference (out of charter — ONNX Runtime handles that)

---

### 🐧 CI / Linux Checklist

**Applies to this repo: PARTIALLY**

Current state:
- ✅ **Platform-conditional tests**: No Skip.* usage detected; all 19 xUnit tests are platform-agnostic.
- ✅ **Cross-platform file chars**: ModelDownloader uses Path.DirectorySeparatorChar correctly.
- ⚠️ **Publish workflow git tag format**: `publish.yml` strips leading `v` with `${VERSION#v}` but does NOT handle stray `.` (e.g., `v.1.0.0 → .1.0.0`). No validation step. **Low risk** because tags are created by Bruno manually.
- ❌ **No explicit Linux CI**: CI runs only on ubuntu-latest; no explicit Windows validation.

**In Scope (Phase 3 — CI/Linux, deferred to next sprint):**
1. **Add git tag format validation** to publish.yml (reject `v.` prefix, require `v\d+\.\d+\.\d+` format)
2. **Add Windows CI** (matrix: ubuntu-latest, windows-latest) for publish workflow

**Out of Scope (not needed yet):**
- SkippableFact (tests don't use Skip.*; not applicable)

---

## Work Items (Priority Order)

### Phase 1 — Security (NOW)

| ID | Title | Description | Assignee | Est. Size | Related Code |
|----|-------|-------------|----------|-----------|--------------|
| SEC-1 | Input validation on TtsPipeline.SynthesizeAsync | Add text length limit (10k chars), null check, encoding validation (UTF-8) on public API entry point. Must raise ArgumentException. | Neo | Medium | Core/TtsPipeline.cs, Web/TtsPipelineService.cs |
| SEC-2 | Path traversal validation in VoiceCloningDownloader | Reject any relativePath containing `..` or absolute paths. Validate before Path.Combine. | Neo | Small | VoiceCloning/Pipeline/VoiceCloningDownloader.cs |
| SEC-3 | ONNX/NPY file size pre-checks | Before loading ONNX sessions or NPY files, check file size is within expected range (e.g., onnx < 2GB, npy < 500MB). | Neo | Small | Core/Models/LanguageModel.cs, Core/Models/NpyReader.cs |
| SEC-4 | Document HTTPS enforcement | Add comment/code review note ensuring all HuggingFace URL constructs use `https://` scheme. VoiceCloningDownloader already enforces this; document why. | Morpheus (review) | Trivial | VoiceCloning/Pipeline/VoiceCloningDownloader.cs |

### Phase 2 — Performance (NEXT SPRINT)

| ID | Title | Description | Assignee | Est. Size | Related Code |
|----|-------|-------------|----------|-----------|--------------|
| PERF-1 | Optimize SampleToken with top-K heap | Replace LanguageModel.SampleToken() LINQ OrderByDescending().Take(topK) with custom min-heap. Reduce from O(n log n) to O(n log k). | Neo | Medium | Core/Models/LanguageModel.cs |
| PERF-2 | ArrayPool for temporary embeddings | Use ArrayPool<float> for cpEmbedBuf, flatCpEmbeds, nextInputBuf in LanguageModel decode loop. Measure allocation churn. | Neo | Medium | Core/Models/LanguageModel.cs |
| PERF-3 | Add BenchmarkDotNet suite | Create ElBruno.QwenTTS.Benchmarks project with latency/allocation benchmarks for GenerateInternal, SampleToken, SpeakerEncoder.Encode. | Tank | Medium | (new project) |
| PERF-4 | TensorPrimitives for softmax | If profiling shows softmax is hot, migrate to System.Numerics.Tensors.TensorPrimitives. Requires investigation. | Trinity (research) | Small | Core/Models/LanguageModel.cs |

### Phase 3 — CI/Linux (LATER)

| ID | Title | Description | Assignee | Est. Size | Related Code |
|----|-------|-------------|----------|-----------|--------------|
| CI-1 | Validate git tag format in publish.yml | Reject tags not matching `v\d+\.\d+\.\d+` (regex). Fail fast with clear error message before build. | Switch | Small | .github/workflows/publish.yml |
| CI-2 | Add Windows CI to publish workflow | Run publish job on matrix: [ubuntu-latest, windows-latest] to catch platform-specific build failures. | Switch | Small | .github/workflows/publish.yml |

---

## Scope Decision

**Phase 1 (NOW): Security work (4 items, medium effort)**
- **Rationale**: Security > performance in prioritization. Core library is on NuGet; public APIs must be hardened. SEC-1 (input validation) blocks VoiceCloning release; SEC-2/3 (path/file validation) prevent supply-chain attacks.
- **Blocker removal**: Unblocks Bruno for immediate release of voice cloning feature.
- **Commitment**: Complete by end of sprint.

**Phase 2 (NEXT SPRINT): Performance work (4 items)**
- **Rationale**: Defer until after Phase 1 lands. Performance optimizations don't block features; they reduce latency/memory. SampleToken top-K is highest ROI.
- **Measurement**: Require benchmarks before/after to prove gains.

**Phase 3 (LATER): CI work (2 items)**
- **Rationale**: Low risk (manual tag creation; no SkippableFact issues). Defer to ensure CI doesn't become bottleneck. Windows validation is nice-to-have, not critical.

---

## Key Decisions

1. **No scope for Model Integrity (SHA-256 hashing, size limits)** — deferred to v1.1.0 audit. Current design trusts HuggingFace HTTPS.
2. **Input validation is blocking for voice cloning** — SEC-1 must land before Web feature goes live.
3. **Performance optimization uses measurements, not guesses** — benchmarks are mandatory for PERF items.
4. **Neo owns security fixes** — Neo reviews all related code and owns the implementation.
5. **Tank validates all performance benchmarks** — requires formal BenchmarkDotNet results before merge.

---

## Next Steps for Squad

1. **Squad (Coordinator)**: Create GitHub issues for each Phase 1 work item (SEC-1 through SEC-4).
2. **Neo**: Begin SEC-1 immediately (text validation on TtsPipeline). Code review with Morpheus.
3. **Morpheus**: Will review all security code changes before merge to ensure coverage.
4. **Tank**: Prepare test strategy for SEC-1 (edge cases: empty string, max length, invalid UTF-8).

---

## Appendix: Reference Materials

- **LocalEmbeddings audit**: https://github.com/elbruno/elbruno.localembeddings/issues/38
- **LocalEmbeddings PR #37**: https://github.com/elbruno/elbruno.localembeddings/pull/37 (security fixes)
- **CI lessons skill** (LocalEmbeddings): `.squad/skills/ci-linux-test-failures/SKILL.md`
