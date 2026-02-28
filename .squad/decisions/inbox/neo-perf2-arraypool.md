# Decision: ArrayPool Adoption in ONNX Inference (PERF-2)

**Date:** 2026-02-28  
**By:** Neo (.NET Developer)  
**Status:** ✅ Complete  
**Closes:** Issue #22 PERF-2

## What

Applied `ArrayPool<T>.Shared` to hot allocation paths in `LanguageModel.cs` ONNX inference loops. Replaces per-iteration heap allocations with pooled array reuse to reduce GC pressure and latency variance during real-time TTS synthesis.

## Optimized Hot Paths

### 1. Prefill Stage (lines 128-161)
**Before:** Heap-allocated flat buffers for ONNX input tensors (embeddings, attention mask, position IDs)  
**After:** Rented from ArrayPool, wrapped in try-finally, returned after ONNX session completes

```csharp
var flatEmbeds = ArrayPool<float>.Shared.Rent(embedSize);
var flatMask = ArrayPool<long>.Shared.Rent(maskSize);
var flatPosIds = ArrayPool<long>.Shared.Rent(posSize);
try {
    // ONNX prefill
} finally {
    ArrayPool<float>.Shared.Return(flatEmbeds);
    ArrayPool<long>.Shared.Return(flatMask);
    ArrayPool<long>.Shared.Return(flatPosIds);
}
```

### 2. Decode Loop (lines 184-314)
**Before:** Per-step heap allocations for attention mask (`newAttentionMask`, `flatDecodeMask`) and CP inputs (`cpInputs` 3D array, `flatCpEmbeds`)  
**After:** Rented large buffers once before loop, reused per-step with `.AsMemory()` slicing

```csharp
var pooledMask = ArrayPool<long>.Shared.Rent(prefillLen + maxNewTokens);
var pooledCpInputs = ArrayPool<float>.Shared.Rent(2 * 1024);
try {
    for (int step = 0; step < maxNewTokens; step++) {
        // Reuse pooledMask for attention, pooledCpInputs for CP embeddings
    }
} finally {
    ArrayPool<long>.Shared.Return(pooledMask);
    ArrayPool<float>.Shared.Return(pooledCpInputs);
}
```

**Inner CP loop:** Dynamically rent `flatCpEmbeds` per group iteration (15× per decode step) with nested try-finally for safety.

### 3. Sampling Methods
**Before:** `new float[vocabSize]` per sampling call (SampleToken: 3072 floats, SampleTokenSimple: 2048 floats)  
**After:** Rent from ArrayPool, wrap entire method in try-finally, return on all paths (including early exit)

```csharp
var probs = ArrayPool<float>.Shared.Rent(vocabSize);
try {
    // Sampling logic
    return sampledToken;
} finally {
    ArrayPool<float>.Shared.Return(probs);
}
```

## Design Rationale

1. **Rent once, reuse many:** Large buffers (mask, CP inputs) rented once before loops to amortize pool overhead. Only small per-iteration buffers (`flatCpEmbeds`) rented dynamically.

2. **Exception safety:** All rentals wrapped in try-finally. Arrays returned even on early loop exit (`break` on `codec_eos`) or exceptions.

3. **Memory slicing:** Use `.AsMemory(0, actualSize)` to pass exact-sized views to ONNX without copying. ArrayPool may return arrays larger than requested.

4. **Zero behavioral change:** Logic remains identical — only allocation strategy differs. All outputs numerically equivalent to heap-allocated version.

## GC Pressure Reduction

- **Prefill:** ~10KB-50KB per synthesis (3 tensors) → pooled (1× allocation amortized across requests)
- **Decode loop:** ~4KB-12KB per step × 2048 max steps = 8MB-24MB → 2 rentals amortized across loop
- **Code Predictor:** ~2KB per group × 15 groups × 2048 steps = 60MB total → now pooled (~30 rentals per step)
- **Sampling:** ~12KB per call × 2048 calls = 24MB → pooled

**Total potential GC reduction:** ~100MB per 2048-token synthesis (assuming full-length generation).

## Testing & Validation

✅ **All 60 tests pass** (50 Core + 10 VoiceCloning) in both Debug and Release modes  
✅ **Zero warnings/errors** across 7 projects  
✅ **Benchmarking:** Ready for Tank's GC profiling to quantify latency variance reduction

## Alternatives Considered

1. **Stackalloc:** Limited to small buffers (≤ few KB). Decode loop buffers (2048+ elements) would risk stack overflow.
2. **Manual buffer reuse:** Error-prone (forget to reset state between iterations). ArrayPool handles pooling/clearing automatically.
3. **MemoryPool<T>:** More control but higher complexity. ArrayPool is sufficient for fixed-size temp buffers.

## Future Work

- **Benchmark Gen2 collections:** Measure GC pause reduction under load (Tank will profile with `dotnet-counters` / BenchmarkDotNet)
- **Profile pool contention:** If concurrent synthesis requests hit pool limits, consider dedicated ArrayPool instances per TtsPipeline
- **KV cache pooling:** Consider pooling past_keys/past_values arrays (largest allocations ~28MB for 2048 tokens). Requires careful lifetime management since they grow per-step.

## Files Modified

- `src/ElBruno.QwenTTS.Core/Models/LanguageModel.cs` — Added `using System.Buffers`, applied ArrayPool to prefill/decode/sampling

## References

- Issue #22 PERF-2: "Reuse pooled arrays in ONNX inference loops"
- ArrayPool<T> docs: https://learn.microsoft.com/dotnet/api/system.buffers.arraypool-1
- Branch: `squad/perf-2-arraypool`
