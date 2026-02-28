# Decision: PERF-1 Top-K Heap Speaker Similarity Search

**Date:** 2026-02-28  
**Author:** Neo (.NET Developer)  
**Status:** ✅ Implemented  
**Branch:** squad/perf-1-topk-heap  
**Closes:** Issue #22 PERF-1

## Context

Issue #22 (audit lessons from LocalEmbeddings v1.1.0) identified Top-K heap optimization as a performance improvement for ranking/similarity search. The task was to replace linear O(n) speaker lookup with O(k log n) heap-based Top-K selection.

Current codebase uses:
- **CustomVoice model**: Hardcoded speaker name → ID lookups (no similarity search)
- **VoiceClone model**: Direct ECAPA-TDNN embedding injection (no similarity search)

**Decision**: Implement Top-K heap proactively as a foundation for future "find similar voices" features, avoiding technical debt.

## Implementation

### SpeakerSimilaritySearch Class
Created static utility class with `FindTopK()` method:
- **Input**: Query embedding (ReadOnlySpan<float>), reference collection (IEnumerable<(string id, float[] embedding)>), K (int)
- **Output**: SpeakerMatch[] ordered by similarity (highest first)
- **Algorithm**: Min-heap maintains only top K similarities; O(n log k) time complexity

### MinHeap Internal Class
Binary min-heap implementation:
- **Insert()**: O(log k) — only accepts values > minimum when heap is full
- **BubbleUp/BubbleDown**: Standard heap operations for maintaining heap property
- **ExtractAll()**: Returns results in descending order (extract min repeatedly)

### SIMD Acceleration
Uses `System.Numerics.Tensors.TensorPrimitives`:
- **TensorPrimitives.Dot()**: Cosine similarity computation (SIMD-accelerated dot product)
- **TensorPrimitives.Norm()**: L2 norm for vector normalization
- **TensorPrimitives.Divide()**: Unit vector computation (v / ||v||)

### Normalization Strategy
Automatic L2 normalization of query and reference embeddings:
- Handles unnormalized inputs gracefully
- Zero vector edge case: Copy as-is (norm < 1e-8f)
- Ensures cosine similarity is in [-1, 1] range

### EmbeddingStore Integration
Added two new public methods:
1. **GetSpeakerEmbedding(int speakerId)**: Retrieves single speaker embedding (1024-dim) from talker_codec_embedding matrix
2. **GetAllSpeakerEmbeddings()**: Yields (name, embedding) tuples for all speakers — optimized for Top-K iteration

## Performance Characteristics

- **Time complexity**: O(n log k) vs O(n log n) for full sort
- **Space complexity**: O(k) heap vs O(n) for array sort
- **Benchmark baseline**: 7.11 ms average for Top-10 from 1000 speakers (1024-dim, 100 iterations)

### Complexity Analysis
For n=1000 speakers, k=10:
- **Min-heap (this impl)**: 1000 × log₂(10) = ~3,322 operations
- **Full sort + take**: 1000 × log₂(1000) = ~9,966 operations
- **Speedup**: ~3× theoretical improvement

## Test Coverage

**11 new tests** in SpeakerSimilaritySearchTests.cs:
1. **FindTopK_IdenticalVector_ReturnsExactMatch**: Validates exact match returns similarity ~1.0
2. **FindTopK_ThreeResults_ReturnsInDescendingOrder**: Ensures results sorted by similarity (highest first)
3. **FindTopK_MoreResultsThanAvailable_ReturnsAll**: Handles k > n gracefully
4. **FindTopK_LargeCollection_MaintainsCorrectTopK**: 1000 speakers, top 5 correctly identified
5. **FindTopK_NormalizedAndUnnormalized_ProduceSameRanking**: Normalization invariance
6. **FindTopK_ZeroVector_HandlesGracefully**: No crash on degenerate input
7. **FindTopK_DimensionMismatch_ThrowsArgumentException**: Input validation
8. **FindTopK_InvalidK_ThrowsArgumentException**: k ≤ 0 rejected
9. **FindTopK_EmptyReferences_ReturnsEmptyArray**: Empty collection handling
10. **FindTopK_HighDimensionalEmbeddings_WorksCorrectly**: 1024-dim realistic test
11. **Benchmark_TopK_1000Speakers_K10**: Baseline performance measurement

**Total Core tests**: 50 passing (39 existing + 11 new PERF-1)  
**Regression testing**: All existing tests pass — no breaking changes

## Future Use Cases

This optimization enables:
1. **Voice similarity search**: Given a cloned voice embedding, find closest built-in speakers
2. **Speaker recommendation**: "This cloned voice sounds like Ryan + Dylan"
3. **Voice morphing**: Blend top-K similar speakers for new voice characteristics
4. **Quality metrics**: Measure how unique/generic a cloned voice is (distance to nearest built-in)

## Design Decisions

### Why min-heap instead of full sort?
- **Efficiency**: O(n log k) vs O(n log n) — significant for large n, small k
- **Memory**: O(k) vs O(n) — important when n is large (e.g., 10,000 reference speakers)
- **Streaming**: Heap supports online/streaming similarity search (references don't need to fit in memory)

### Why TensorPrimitives instead of manual loops?
- **SIMD**: Automatic vectorization for dot product, norm, divide
- **Correctness**: Battle-tested library implementation (no manual floating-point bugs)
- **Future-proof**: Benefits from .NET runtime SIMD improvements automatically

### Why cosine similarity?
- **Standard**: Industry-standard for high-dimensional embedding similarity
- **Normalized**: Magnitude-invariant (only cares about direction)
- **Interpretable**: Range [-1, 1] with clear semantics (1 = identical, 0 = orthogonal, -1 = opposite)

### Why proactive implementation?
- **Avoids technical debt**: Implementing efficient algorithm now vs refactoring later
- **Enables exploration**: Team can prototype "find similar voices" features immediately
- **ROI**: Minimal implementation cost (~300 LOC + tests) for 3× speedup potential

## Files Changed

**Created:**
- `src/ElBruno.QwenTTS.Core/Models/SpeakerSimilaritySearch.cs` (172 lines)
- `src/ElBruno.QwenTTS.Core.Tests/SpeakerSimilaritySearchTests.cs` (260 lines)

**Modified:**
- `src/ElBruno.QwenTTS.Core/Models/EmbeddingStore.cs` (+29 lines: GetSpeakerEmbedding, GetAllSpeakerEmbeddings)

**Build status**: 0 warnings, 0 errors  
**Test status**: 50/50 Core tests passing

## Recommendation

✅ **Merge to main**  
This is a clean, isolated optimization with comprehensive tests and no breaking changes. Enables future voice similarity features without refactoring.
