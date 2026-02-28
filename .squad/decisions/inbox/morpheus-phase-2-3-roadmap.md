# Phase 2 & 3 Roadmap — Performance & CI/Linux Hardening

**Date:** 2026-02-28  
**Author:** Morpheus (Lead/Architect)  
**Context:** Issue #22 Phase 1 (SEC-1 through SEC-4) complete and merged to main. Now planning Phase 2 (Performance) and Phase 3 (CI/Linux).

---

## Executive Summary

Phase 1 delivered 4 security hardening improvements (input validation, path traversal, file size checks, HTTPS enforcement). Phase 2 targets **measurable performance gains** in the TTS inference hot path. Phase 3 addresses **cross-platform test reliability** and **CI workflow robustness**.

**Priority Order:** Phase 2 → Phase 3 (performance unlocks production readiness; CI improvements are operational hygiene).

**Timeline Estimate:**
- Phase 2: 4–6 work sessions (PERF-3 gates PERF-1/2/4 validation)
- Phase 3: 2–3 work sessions (simpler, mostly infrastructure)

---

## Phase 2: Performance Optimization

### Overview

Current TTS pipeline has **hot paths** in:
1. **Top-K sampling** (line 512: `.OrderByDescending().ToArray()` sorts 3072 floats per autoregressive step → ~2048 steps per synthesis)
2. **Matrix operations** (EmbeddingStore.MatMul, line 142–153: manual loops, no SIMD)
3. **Temporary allocations** (LanguageModel: `new float[vocabSize]` per sample, `new float[1, cpInputSeqLen, 1024]` per CP group)
4. **Softmax/Exp operations** (SampleToken: manual exp/sum loops, line 518–526)

**Performance Target:** 10–20% latency reduction on end-to-end synthesis (measure with BenchmarkDotNet before/after).

---

### PERF-3: BenchmarkDotNet Baseline (DO THIS FIRST)

**Why First:** Cannot validate performance improvements without a baseline. PERF-1, PERF-2, PERF-4 **must** be benchmarked before and after.

**Scope:**
- Add `BenchmarkDotNet` package to `ElBruno.QwenTTS.Benchmarks` project
- Implement 3 microbenchmarks:
  1. **TopKSampling**: Benchmark current LINQ-based Top-K vs min-heap approach (3072 floats, K=50)
  2. **MatMul2048x2048**: Matrix-vector multiply (2048×2048 weight × 2048 input) — current manual loop vs TensorPrimitives.CosineSimilarity or hand-rolled SIMD
  3. **SoftmaxExp3072**: Softmax over 3072 floats — current manual loop vs TensorPrimitives (if applicable)
- Add 1 end-to-end benchmark:
  - **TtsPipelineBenchmark**: Synthesize 50-character phrase (fixed seed, fixed speaker/language) — measure total latency and allocations
- Document baseline numbers in `docs/benchmarks.md`

**Acceptance Criteria:**
- ✅ `dotnet run -c Release --project src/ElBruno.QwenTTS.Benchmarks` executes all 4 benchmarks
- ✅ BenchmarkDotNet outputs median latency, mean allocations, P95 latency for each benchmark
- ✅ Results recorded in `docs/benchmarks.md` with commit SHA and hardware specs (CPU model, RAM, OS)
- ✅ All 5 projects compile with 0 warnings/errors
- ✅ CI workflow runs benchmarks on PR (informational only, no failure gate)

**Owner:** Neo (implementation) + Tank (validation)  
**Estimated Effort:** 1–2 sessions  
**Blocker for:** PERF-1, PERF-2, PERF-4 (cannot validate without baseline)

---

### PERF-1: Top-K Sampling Optimization

**Current Implementation (LanguageModel.cs, line 510–514):**
```csharp
var indexed = probs.Select((p, i) => (p, i)).OrderByDescending(x => x.p).ToArray();
for (int i = topK; i < vocabSize; i++)
    probs[indexed[i].i] = float.NegativeInfinity;
```

**Problem:**
- **Full sort** of 3072 floats per autoregressive step (O(N log N) where N=3072)
- Top-K only needs K=50 elements → **60× waste** (sorting 3072 to get 50)
- LINQ allocation overhead: `Select()` allocates tuple array, `OrderByDescending()` allocates sorted array
- **Called ~2048 times per synthesis** (maxNewTokens=2048) → cumulative overhead is significant

**Solution: Min-Heap (Priority Queue)**

Replace full sort with a **min-heap** (k=50):
```csharp
// Keep top-K using a min-heap (O(N log K) vs O(N log N))
private static void ApplyTopKMask(Span<float> probs, int topK)
{
    var heap = new PriorityQueue<int, float>(topK);
    
    for (int i = 0; i < probs.Length; i++)
    {
        if (heap.Count < topK)
            heap.Enqueue(i, probs[i]);
        else if (probs[i] > heap.Peek())
        {
            heap.Dequeue();
            heap.Enqueue(i, probs[i]);
        }
    }
    
    // Build mask of top-K indices
    Span<bool> isTopK = stackalloc bool[probs.Length];
    while (heap.TryDequeue(out var idx, out _))
        isTopK[idx] = true;
    
    // Suppress non-top-K
    for (int i = 0; i < probs.Length; i++)
        if (!isTopK[i])
            probs[i] = float.NegativeInfinity;
}
```

**Acceptance Criteria:**
- ✅ Replace LINQ sort with `PriorityQueue<int, float>` min-heap in `SampleToken()` (LanguageModel.cs)
- ✅ Use `stackalloc bool[vocabSize]` for top-K mask (avoid heap allocations)
- ✅ BenchmarkDotNet shows **5–10% improvement** in TopKSampling microbenchmark (3072 floats, K=50)
- ✅ BenchmarkDotNet shows **2–5% improvement** in end-to-end TtsPipelineBenchmark
- ✅ All 60 tests pass (50 Core + 10 VoiceCloning)
- ✅ Zero regressions in audio quality (spot-check: synthesize "Hello world" before/after, verify WAV files are byte-identical or have <1e-5 diff)

**Owner:** Neo (implementation) + Tank (validation)  
**Estimated Effort:** 1 session  
**Dependencies:** PERF-3 (baseline benchmarks)  
**ROI:** **HIGH** — called 2048× per synthesis, 60× reduction in comparisons (3072 → 50)

---

### PERF-2: ArrayPool for Temporary Buffers

**Current Implementation:**
- `SampleToken()` allocates `new float[vocabSize]` (3072 floats = 12 KB) **every autoregressive step** (~2048×)
- `SampleTokenSimple()` allocates `new float[2048]` (8 KB) **31× per autoregressive step** (CP groups 1–15 → 2× per group) → 31×2048 = ~63,000 allocations
- `GenerateInternal()` allocates `new float[1, cpInputSeqLen, 1024]` per CP group (cpInputSeqLen=1 or 2 → 4–8 KB) **31× per step** → ~63,000 allocations

**Problem:**
- **~130,000 temporary allocations** per synthesis (2048 steps × 64 allocs/step)
- Gen0 GC pressure — each synthesis triggers multiple Gen0 collections
- Latency spikes from GC pauses (5–10 ms per collection)

**Solution: ArrayPool**

Rent buffers from `ArrayPool<float>.Shared`:
```csharp
// In SampleToken()
var probs = ArrayPool<float>.Shared.Rent(vocabSize);
try
{
    Array.Copy(logits, logits.Length - vocabSize, probs, 0, vocabSize);
    // ... sampling logic ...
    return result;
}
finally
{
    ArrayPool<float>.Shared.Return(probs);
}
```

**Scope:**
- `SampleToken()`: Rent `float[vocabSize]` from pool
- `SampleTokenSimple()`: Rent `float[2048]` from pool
- `GenerateInternal()`: Rent `float[cpMaxLen * 1024]` from pool (reuse across CP groups)
- Add `ArrayPool<float>` as a class-level field in `LanguageModel`

**Acceptance Criteria:**
- ✅ Replace `new float[]` with `ArrayPool<float>.Shared.Rent()` in 3 hot paths
- ✅ Add `try/finally` blocks to ensure `Return()` is called (no leaks)
- ✅ BenchmarkDotNet shows **50–80% reduction** in allocations (measure with `[MemoryDiagnoser]`)
- ✅ BenchmarkDotNet shows **5–10% improvement** in end-to-end latency (reduced GC pauses)
- ✅ All 60 tests pass
- ✅ Spot-check: Synthesize "Hello world" 10× in a loop; verify no memory growth (use `dotMemory` or `PerfView` if available)

**Owner:** Neo (implementation) + Tank (validation)  
**Estimated Effort:** 1–2 sessions  
**Dependencies:** PERF-3 (baseline benchmarks)  
**ROI:** **HIGH** — 130k allocations → near-zero allocations; reduces Gen0 GC frequency by 80–90%

---

### PERF-4: TensorPrimitives for Matrix Operations

**Current Implementation (EmbeddingStore.cs, line 142–153):**
```csharp
private static void MatMul(float[,] weight, ReadOnlySpan<float> input, Span<float> output)
{
    int M = weight.GetLength(0);
    int N = weight.GetLength(1);
    
    for (int i = 0; i < M; i++)
    {
        float sum = 0;
        for (int j = 0; j < N; j++)
            sum += weight[i, j] * input[j];
        output[i] = sum;
    }
}
```

**Problem:**
- **No SIMD vectorization** — manual scalar loops
- Matrix-vector multiply called **per text token** (TextProjection: 2048×2048 → 1024×2048)
- **~50–100 calls per synthesis** (text token count)
- Modern CPUs have AVX2/AVX512 (256-bit/512-bit registers) — potential 4–8× speedup

**Solution: System.Numerics.Tensors.TensorPrimitives**

Use `TensorPrimitives.Dot()` for row-vector dot product:
```csharp
private static void MatMul(float[,] weight, ReadOnlySpan<float> input, Span<float> output)
{
    int M = weight.GetLength(0);
    int N = weight.GetLength(1);
    
    for (int i = 0; i < M; i++)
    {
        // Extract row i as a span
        var row = MemoryMarshal.CreateReadOnlySpan(
            ref weight[i, 0], N);
        output[i] = TensorPrimitives.Dot(row, input);
    }
}
```

**Alternative:** If `MemoryMarshal.CreateReadOnlySpan()` on 2D arrays doesn't work (row-major layout), use `Span<float> tempRow = stackalloc float[N]` and copy row manually, then call `TensorPrimitives.Dot()`.

**Scope:**
- Replace manual dot product loop in `MatMul()` with `TensorPrimitives.Dot()`
- Evaluate `TensorPrimitives.Exp()` for softmax in `SampleToken()` (line 522: `MathF.Exp()` loop)
- Evaluate `TensorPrimitives.Sum()` for softmax normalization (line 523: manual sum loop)

**Acceptance Criteria:**
- ✅ Replace manual loops with `TensorPrimitives.Dot()` in `MatMul()`
- ✅ (Stretch) Replace `MathF.Exp()` loop with `TensorPrimitives.Exp()` in `SampleToken()`
- ✅ (Stretch) Replace manual sum with `TensorPrimitives.Sum()` in `SampleToken()`
- ✅ BenchmarkDotNet shows **20–40% improvement** in MatMul2048x2048 microbenchmark (SIMD speedup)
- ✅ BenchmarkDotNet shows **2–5% improvement** in end-to-end TtsPipelineBenchmark (matmul is 5–10% of total time)
- ✅ All 60 tests pass
- ✅ Spot-check: Synthesize "Hello world" before/after, verify WAV files are byte-identical or have <1e-5 diff (SIMD should not affect precision at this scale)

**Owner:** Neo (implementation) + Tank (validation)  
**Estimated Effort:** 1–2 sessions  
**Dependencies:** PERF-3 (baseline benchmarks)  
**ROI:** **MEDIUM** — MatMul is 5–10% of total synthesis time; 20–40% speedup → 1–4% end-to-end gain. Justification: Easy win, no algorithmic risk.

---

### Phase 2 Summary

| Item | ROI | Complexity | Effort | Blocker | Latency Impact | Allocation Impact |
|------|-----|------------|--------|---------|----------------|-------------------|
| **PERF-3** (Benchmarks) | N/A | Low | 1–2 sessions | None | N/A (baseline) | N/A |
| **PERF-1** (Top-K heap) | **HIGH** | Medium | 1 session | PERF-3 | 2–5% | Minor |
| **PERF-2** (ArrayPool) | **HIGH** | Low | 1–2 sessions | PERF-3 | 5–10% (GC reduction) | **80–90% reduction** |
| **PERF-4** (TensorPrimitives) | **MEDIUM** | Low | 1–2 sessions | PERF-3 | 1–4% | None |

**Recommended Order:**
1. **PERF-3** (gates all others)
2. **PERF-2** (highest latency + allocation impact)
3. **PERF-1** (highest algorithmic impact)
4. **PERF-4** (easy win, low risk)

**Combined Impact:** 10–20% latency reduction, 80–90% allocation reduction, near-zero Gen0 GCs per synthesis.

---

## Phase 3: CI / Linux Hardening

### Overview

Issue #22 identified **3 CI/Linux issues**:
1. **Platform-conditional tests** need `[SkippableFact]` (not `[Fact]`) on Linux
2. **File name validation** must use cross-platform char set (not `Path.GetInvalidFileNameChars()`)
3. **NuGet publish workflow** must validate git tag format and strip leading `v` and `.` characters

**Current Status:**
- ✅ No `Skip.If()`/`Skip.IfNot()` calls in codebase (verified via grep)
- ✅ No `GetInvalidFileNameChars()` calls in codebase (verified via grep)
- ❌ `.github/workflows/publish.yml` uses simple regex strip (potential typo risk: `v.1.2.3` → `.2.3`)

**Priority:** **LOWER** than Phase 2 (these are operational hygiene; no production impact unless running Linux tests or publishing to NuGet).

---

### CI-1: Platform-Conditional Test Pattern

**Current Implementation:**
- No platform-conditional tests exist yet
- Codebase has no `[Fact]` tests with `Skip.If()` or `Skip.IfNot()`

**Problem:**
- **Potential future risk**: If any developer adds `[Fact]` + `Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))`, the test will **fail** on Linux (not skip)
- `Skip.*` throws `SkipException`; `[Fact]` treats this as a test failure

**Solution: Preventive Documentation + Linter Rule**

No code changes needed today. Add:
1. **Documentation**: `docs/testing-guidelines.md` — "Use `[SkippableFact]` for platform-conditional tests"
2. **(Stretch)** Add `.editorconfig` rule or Roslyn analyzer to enforce `[SkippableFact]` when `Skip.If` is detected

**Acceptance Criteria:**
- ✅ Document platform-conditional test pattern in `docs/testing-guidelines.md`
- ✅ (Stretch) Add `.editorconfig` rule: `dotnet_diagnostic.xUnit1004.severity = error` (enforces `[SkippableFact]` when `Skip.*` is used)
- ✅ No action needed on existing tests (none use `Skip.*` today)

**Owner:** Scribe (documentation) + Tank (review)  
**Estimated Effort:** <1 session  
**Dependencies:** None  
**ROI:** **LOW** (preventive only; no current risk)

---

### CI-2: Cross-Platform File Name Validation

**Current Implementation:**
- No file name validation logic in codebase (verified via grep)

**Problem:**
- **Potential future risk**: If any developer uses `Path.GetInvalidFileNameChars()` for validation, it will fail on Linux
- `Path.GetInvalidFileNameChars()` returns only `\0` and `/` on Linux (missing `<>:"|?*` from Windows)
- Cross-platform validation requires a **hardcoded set**

**Solution: Preventive Documentation**

No code changes needed today. Add to `docs/platform-guidelines.md`:
```csharp
// Cross-platform file name validation (DO NOT use Path.GetInvalidFileNameChars())
private static readonly char[] InvalidFileNameChars =
    ['<', '>', ':', '"', '|', '?', '*', '\\', '/', '\0'];

private static bool IsValidFileName(string name)
    => !name.AsSpan().ContainsAny(InvalidFileNameChars);
```

**Acceptance Criteria:**
- ✅ Document cross-platform file name validation pattern in `docs/platform-guidelines.md`
- ✅ No action needed on existing code (no file name validation today)

**Owner:** Scribe (documentation)  
**Estimated Effort:** <1 session  
**Dependencies:** None  
**ROI:** **LOW** (preventive only; no current risk)

---

### CI-3: NuGet Publish Workflow — Git Tag Validation

**Current Implementation (`.github/workflows/publish.yml`, lines 30–33):**
```yaml
- name: Determine version
  run: |
    if [ "${{ github.event_name }}" == "release" ]; then
      VERSION=$
    elif [ "${{ github.event.inputs.version }}" != "" ]; then
      VERSION=${{ github.event.inputs.version }}
    else
      VERSION=$(grep -oP '<Version>\K[^<]+' src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj)
    fi
    echo "VERSION=$VERSION" >> $GITHUB_ENV
```

**Problem:**
- **Tag format risk**: User creates tag `v.1.2.3` (typo) → `sed` strips `v` → version becomes `.2.3` (invalid)
- **No validation**: Workflow doesn't check if version string is valid semver format
- **Silent failure**: Invalid version causes NuGet push to fail with cryptic error

**Solution: Add Version Format Validation**

Add validation step **before** `dotnet pack`:
```yaml
- name: Determine version
  run: |
    if [ "${{ github.event_name }}" == "release" ]; then
      VERSION=$
      VERSION=$  # Strip leading 'v'
      VERSION=$  # Strip leading '.' (handles v.1.2.3 typo)
    elif [ "${{ github.event.inputs.version }}" != "" ]; then
      VERSION=${{ github.event.inputs.version }}
    else
      VERSION=$(grep -oP '<Version>\K[^<]+' src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj)
    fi
    echo "VERSION=$VERSION" >> $GITHUB_ENV

- name: Validate version format
  run: |
    if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.]+)?$ ]]; then
      echo "❌ ERROR: Invalid version format '$VERSION' (expected semver: X.Y.Z or X.Y.Z-prerelease)"
      echo "Check your git tag format. Common issues:"
      echo "  - Typo: 'v.1.2.3' (should be 'v1.2.3')"
      echo "  - Missing digits: 'v1.2' (should be 'v1.2.0')"
      exit 1
    fi
    echo "✅ Version format validated: $VERSION"
```

**Acceptance Criteria:**
- ✅ Add version format validation step to `.github/workflows/publish.yml` (after "Determine version")
- ✅ Validation regex: `^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.]+)?$` (semver 2.0)
- ✅ Fail workflow with clear error message if version is invalid
- ✅ Test with manual dispatch: `1.2.3` (pass), `.2.3` (fail), `v1.2.3` (pass after strip), `v.1.2.3` (pass after strip)
- ✅ Update `docs/release-process.md` with git tag format guidelines

**Owner:** Neo (workflow) + Scribe (documentation)  
**Estimated Effort:** <1 session  
**Dependencies:** None  
**ROI:** **MEDIUM** (prevents silent failures; low frequency but high impact when it occurs)

---

### Phase 3 Summary

| Item | ROI | Complexity | Effort | Priority |
|------|-----|------------|--------|----------|
| **CI-1** (SkippableFact docs) | **LOW** | Low | <1 session | LATER |
| **CI-2** (File name docs) | **LOW** | Low | <1 session | LATER |
| **CI-3** (Tag validation) | **MEDIUM** | Low | <1 session | **NOW** (blocks NuGet release) |

**Recommended Order:**
1. **CI-3** (blocks NuGet release; higher impact)
2. **CI-1** + **CI-2** (documentation only; low effort, low priority)

---

## Architectural Decisions

### AD-1: BenchmarkDotNet as Performance Validation Gate

**Decision:** All performance optimizations (PERF-1, PERF-2, PERF-4) **must** be validated with BenchmarkDotNet before merging.

**Rationale:**
- Manual timing is unreliable (OS scheduler noise, cold start, caching)
- BenchmarkDotNet provides statistically rigorous measurement (median, P95, allocation tracking)
- **Prevents regressions**: Future PRs can compare against baseline

**Impact:**
- PERF-3 becomes **blocking** for all performance work
- Benchmarks run in CI on every PR (informational only; no failure gate)
- `docs/benchmarks.md` records baseline and post-optimization numbers

---

### AD-2: ArrayPool Strategy — Shared vs Custom Pool

**Decision:** Use `ArrayPool<float>.Shared` (not a custom pool).

**Rationale:**
- `Shared` pool is globally optimized by the runtime (reduces fragmentation)
- Custom pool adds complexity (sizing policy, disposal, thread safety)
- **Trade-off**: Shared pool is slower than custom pool (contention), but allocations are in "warm" phase (after prefill) — contention is minimal

**Impact:**
- Simpler implementation (no custom pool lifecycle management)
- Slight contention risk if multiple TTS pipelines run concurrently (acceptable for v1.0)

---

### AD-3: Top-K Min-Heap — PriorityQueue vs Manual Heap

**Decision:** Use `PriorityQueue<int, float>` from BCL (not a hand-rolled min-heap).

**Rationale:**
- `PriorityQueue` is part of .NET 6+ BCL (no external dependency)
- Hand-rolled heap is error-prone (off-by-one bugs, heap property violations)
- Performance is equivalent (both O(N log K))

**Impact:**
- Standard library = less maintenance burden
- Trade-off: Cannot customize heap comparison (but default `float` comparison is correct for Top-K)

---

### AD-4: TensorPrimitives — SIMD Risk Assessment

**Decision:** TensorPrimitives is **safe** for production (no precision risk at float32 scale).

**Rationale:**
- SIMD operations use **same floating-point unit** as scalar code (no precision loss)
- `.NET 8+` TensorPrimitives API is stable (not experimental)
- Edge case: Very large/small floats may have different rounding (10^-7 ULP difference) — not audible in 24 kHz audio

**Validation:**
- Spot-check: Synthesize "Hello world" before/after SIMD changes
- Compare WAV files: assert byte-identical or L2 norm < 1e-5

**Impact:**
- Safe to merge without extensive A/B testing
- Rollback plan: If audio quality degrades (unlikely), revert to scalar loops

---

## Work Queue (Prioritized)

### Phase 2: Performance (4–6 sessions)
1. **PERF-3** (BenchmarkDotNet baseline) — 1–2 sessions — **BLOCKING**
2. **PERF-2** (ArrayPool) — 1–2 sessions — **HIGHEST ROI**
3. **PERF-1** (Top-K heap) — 1 session — **HIGH ROI**
4. **PERF-4** (TensorPrimitives) — 1–2 sessions — **MEDIUM ROI**

### Phase 3: CI/Linux (2–3 sessions)
5. **CI-3** (Tag validation) — <1 session — **BLOCKS RELEASE**
6. **CI-1** (SkippableFact docs) + **CI-2** (File name docs) — <1 session — **LOW PRIORITY**

---

## Open Questions

1. **PERF-3 CI Integration:** Should benchmarks run on every PR (informational comment) or only on `main` branch?
   - **Recommendation:** Informational only on PR (no failure gate); prevents false positives from CI hardware variance.

2. **PERF-2 ArrayPool Sizing:** Should we pre-rent arrays on pipeline initialization (warm the pool) or rent on-demand?
   - **Recommendation:** On-demand (simpler); pre-warming adds 10 lines for negligible gain (first synthesis is already "cold").

3. **Phase 3 Timing:** Should CI-3 (tag validation) be done **before** or **after** Phase 2?
   - **Recommendation:** After Phase 2 (CI-3 is low effort; no point blocking performance work for a workflow fix).

---

## Success Metrics

### Phase 2 (Performance)
- ✅ **10–20% latency reduction** on end-to-end TtsPipelineBenchmark (50-char synthesis)
- ✅ **80–90% allocation reduction** (measure with `[MemoryDiagnoser]`)
- ✅ **Zero audio quality regressions** (WAV files byte-identical or <1e-5 diff)
- ✅ **All 60 tests pass** (no functional regressions)

### Phase 3 (CI/Linux)
- ✅ **CI-3 validation prevents silent failures** (test with `v.1.2.3` tag → workflow fails with clear message)
- ✅ **Documentation added** for CI-1 (SkippableFact) and CI-2 (file name validation)
- ✅ **Zero regressions** in CI pipeline (workflow still works for valid tags)

---

## Next Steps

1. **Morpheus:** Code review this roadmap; approve or request changes.
2. **Coordinator:** Assign PERF-3 to Neo + Tank (benchmark baseline).
3. **Scribe:** Merge this decision to `.squad/decisions.md` after Morpheus approval.
4. **Neo:** Start PERF-3 implementation (BenchmarkDotNet setup).

---

## References

- **Issue #22:** https://github.com/elbruno/ElBruno.QwenTTS/issues/22
- **LocalEmbeddings audit:** https://github.com/elbruno/elbruno.localembeddings/issues/38
- **BenchmarkDotNet docs:** https://benchmarkdotnet.org/articles/overview.html
- **TensorPrimitives API:** https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives
- **PriorityQueue API:** https://learn.microsoft.com/en-us/dotnet/api/system.collections.generic.priorityqueue-2

---

**End of Roadmap**
