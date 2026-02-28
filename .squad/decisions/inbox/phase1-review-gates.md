# Phase 1 Code Review Gates — Next Steps

**Prepared by:** Scribe (Orchestration Lead)  
**For:** Morpheus (Code Reviewer)  
**Date:** 2026-02-28  
**Status:** Awaiting review

---

## Executive Summary

Phase 1 security hardening (SEC-1 through SEC-4) is **implementation complete**, **fully tested** (60 tests, 100% passing), and **ready for code review**. This memo documents the review checklist for Morpheus, the decision to keep Issue #22 open, and recommendations for Phase 2/3 planning.

---

## Code Review Checklist for Morpheus

### Security Implementation Review

**SEC-1: Input Validation (TtsPipeline + TtsPipelineService)**
- ✅ Null check implementation (ArgumentNullException.ThrowIfNull)
- ✅ Empty string rejection (ArgumentException)
- ✅ Character-count limit (10,000 chars) — verify reasonableness for TTS use case
- ✅ Placement: Both Core library (NuGet boundary) and Web service (HTTP entry point)
- **Reviewer note:** Confirm validation order (null → empty → length) is correct and cannot be bypassed

**SEC-2: Path Traversal (VoiceCloningDownloader)**
- ✅ ValidateRelativePath() static method isolates validation logic
- ✅ Two independent checks: Path.IsPathRooted() + Contains("..")
- ✅ Called at two points: IsModelDownloaded() and DownloadModelAsync()
- ✅ ExpectedFiles array (hardcoded) ensures only known paths are validated
- **Reviewer note:** Confirm ".." substring check is sufficient (no alternative traversal sequences like "..\\")

**SEC-3: File Size Pre-Checks (LanguageModel, Vocoder, NpyReader)**
- ✅ ONNX limit: 2 GB (provides 1.7× headroom above largest model)
- ✅ NPY limit: 500 MB (provides 2–3× headroom above typical embeddings)
- ✅ Exception type: InvalidOperationException (state error, not argument error)
- ✅ Validation order: Lazy for ONNX (before session creation), eager for NPY (before parsing)
- **Reviewer note:** Verify file size constants are sufficient for future model versions (e.g., larger quantized models)

**SEC-4: HTTPS Enforcement (VoiceCloningDownloader)**
- ✅ Hardcoded scheme: `https://` (non-configurable)
- ✅ Threat model documented in XML comments (ONNX = code execution risk, MITM = critical)
- ✅ No HTTP fallback path
- **Reviewer note:** Confirm scheme constant cannot be overridden by environment or config

### Test Coverage Review

**SEC-1 Tests (9 tests in Sec1ValidationTests.cs)**
- Null input → ArgumentNullException ✅
- Empty string → ArgumentException ✅
- Boundary: 9,999 chars → pass ✅
- Boundary: 10,000 chars → pass ✅
- Boundary: 10,001 chars → fail ✅
- Unicode edge cases (emoji, CJK, Arabic, Cyrillic, Japanese) ✅
- Validation order (null checked before length) ✅
- **Reviewer note:** Ensure tests cover all documented exception types and messages

**SEC-3 Tests (11 tests in Sec3FileSizeTests.cs)**
- NPY: 499 MB → pass ✅
- NPY: 500 MB → pass ✅
- NPY: 500 MB + 1 byte → fail ✅
- NPY: 1 GB → fail ✅
- ONNX: 1.9 GB → pass ✅
- ONNX: 2 GB → pass ✅
- ONNX: 2 GB + 1 byte → fail ✅
- ONNX: 4 GB → fail ✅
- Comparative: ONNX = 4× NPY ✅
- Small files pass both ✅
- Medium files pass ONNX, fail NPY ✅
- **Reviewer note:** Verify streaming I/O strategy avoids memory exhaustion in tests

### Build & Regression Review

- ✅ All 7 projects compile: 0 errors, 0 warnings
- ✅ All 60 tests pass: 50 Core + 10 VoiceCloning (100% success rate)
- ✅ No regressions: Pre-existing tests unaffected
- ✅ Code style: Consistent with existing Core library patterns
- **Reviewer action:** Run `dotnet build` + `dotnet test` locally to confirm zero warnings

---

## Decision: Keep Issue #22 Open

**Recommendation:** KEEP Issue #22 OPEN for Phase 2 (Performance Optimization)

**Rationale:**
1. **Phase 1 (Security) is now complete** — all 4 items (SEC-1/2/3/4) delivered, tested, documented
2. **Phase 2 (Performance) is planned but NOT yet started** — Issue #22 references both Phase 1 and Phase 2 work
3. **Phase 2 has 4 work items** (deferred to next sprint):
   - PERF-1: Optimize SampleToken with top-K heap
   - PERF-2: ArrayPool for temporary embeddings
   - PERF-3: Add BenchmarkDotNet suite
   - PERF-4: TensorPrimitives for softmax
4. **Performance optimizations are blocking for performance tier** (not feature tier) — can ship v1.0 without Phase 2

**Action:**
- Morpheus: After approving Phase 1 code, add comment to Issue #22: "Phase 1 (Security) complete and merged. Deferring Phase 2 (Performance) to next sprint."
- Close this ticket only after Phase 2 is also complete, or if performance optimizations are explicitly deprioritized by Bruno

---

## Phase 2 Planning Recommendations

**When:** Next sprint (after Phase 1 merges)  
**Lead:** Neo (implementation) + Tank (benchmarking)

### Work Items (Priority Order)

| ID | Item | Size | Blocker? | Notes |
|----|----|------|---------|-------|
| PERF-1 | SampleToken top-K heap | Medium | ❌ No | Highest ROI (5–10% latency gain). LINQ sort is O(n log n); heap is O(n log k). |
| PERF-2 | ArrayPool for embeddings | Medium | ❌ No | Reduce allocation churn in decode loop. Measure before/after with BenchmarkDotNet. |
| PERF-3 | BenchmarkDotNet suite | Medium | ⚠️ Needed for PERF-1/2 | Required to prove performance gains. Create `ElBruno.QwenTTS.Benchmarks` project. |
| PERF-4 | TensorPrimitives for softmax | Small | ❌ No | Only if profiling shows softmax is hot. Investigate first. |

### Measurement Strategy

- **Before optimization:** Run benchmarks on current code, capture baseline latency/allocations
- **After each optimization:** Re-run benchmarks, document % improvement
- **Acceptance criteria:** Each optimization must show ≥2% measurable improvement (no speculative changes)

### Phase 3 (CI/Linux) Notes

**Deferred to later sprint.** Two small items:
- CI-1: Validate git tag format in publish.yml (prevent `v.1.0.0` → `.1.0.0` issues)
- CI-2: Add Windows CI to publish workflow (catch platform-specific build failures)

Low priority (manual tag creation, no blocking issues) but recommended before public release.

---

## File Summary

### Core Library Changes (NuGet package)
- `src/ElBruno.QwenTTS.Core/Pipeline/TtsPipeline.cs` — SEC-1 validation
- `src/ElBruno.QwenTTS.Core/Models/LanguageModel.cs` — SEC-3 size checks
- `src/ElBruno.QwenTTS.Core/Models/Vocoder.cs` — SEC-3 size check
- `src/ElBruno.QwenTTS.Core/Models/NpyReader.cs` — SEC-3 size check

### Web App Changes
- `src/ElBruno.QwenTTS.Web/Services/TtsPipelineService.cs` — SEC-1 validation

### Voice Cloning Changes
- `src/ElBruno.QwenTTS.VoiceCloning/Pipeline/VoiceCloningDownloader.cs` — SEC-2 path validation, SEC-4 HTTPS docs

### Test Files (NEW)
- `src/ElBruno.QwenTTS.Core.Tests/Sec1ValidationTests.cs` — 9 SEC-1 validation tests
- `src/ElBruno.QwenTTS.Core.Tests/Sec3FileSizeTests.cs` — 11 SEC-3 size validation tests

### Orchestration & Memory
- `.squad/orchestration-log/2026-02-28T195000Z-neo-sec3.md` — Neo SEC-3 completion
- `.squad/orchestration-log/2026-02-28T195000Z-tank-sec3-validation.md` — Tank SEC-3 validation
- `.squad/orchestration-log/2026-02-28T195000Z-switch-sec4.md` — Switch SEC-4 documentation
- `.squad/log/2026-02-28-phase1-security-complete.md` — Phase 1 session summary
- `.squad/decisions.md` — Merged all decision memos (SEC-1/2/3/4, Phase 1 consolidation)

---

## Next Actions for Morpheus

1. **Code review:** Check all files listed above against security review checklist
2. **Local validation:** Run `dotnet build && dotnet test` to confirm zero warnings
3. **Approval decision:** Comment on PR/branch with approval + any requested changes
4. **Issue #22 comment:** Add "Phase 1 complete" note + defer Phase 2 to next sprint
5. **Merge:** Squash merge to main with commit message including Co-authored-by trailer

---

**Status:** ✅ READY FOR REVIEW  
**Confidence:** HIGH (all items tested, zero warnings, decision documented)
