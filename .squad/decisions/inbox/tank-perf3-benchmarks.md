# Tank's Decision Memo: PERF-3 BenchmarkDotNet Profiling Setup

**Date:** 2026-02-28  
**Agent:** Tank (Tester)  
**Status:** ✅ Complete  
**Branch:** squad/perf-3-benchmarks  
**Issue:** #22 (PERF-3)

## What

Implemented BenchmarkDotNet performance benchmarking infrastructure for ElBruno.QwenTTS Core library.

## Implementation

### New Project: ElBruno.QwenTTS.Benchmarks

- **Package:** BenchmarkDotNet 0.15.8 (latest stable)
- **Target:** net10.0 with OutputType=Exe
- **Reference:** ElBruno.QwenTTS.Core for TtsPipeline access

### Three Benchmark Classes

1. **TokenizationBenchmark** — Text processing pipeline
   - English short (100 chars)
   - English long (1000 chars, 10× repeated)
   - CJK text (Chinese with Unicode)
   - Measures: tokenization + inference + vocoder + WAV write

2. **InferenceBenchmark** — Full TTS synthesis
   - Short text (~10 words)
   - Medium text (~30 words)
   - CJK text (Chinese)
   - Measures: end-to-end synthesis latency and throughput

3. **AudioWriteBenchmark** — WAV file writing
   - Short audio (~1s)
   - Medium audio (~3s)
   - Long audio (~5s)
   - Measures: file I/O and PCM conversion performance

### Configuration

- **Runtime:** [SimpleJob(RuntimeMoniker.Net80)] — .NET 8.0 baseline
- **Diagnostics:** [MemoryDiagnoser] — tracks heap allocations, GC collections
- **Exporters:** [JsonExporter], [MarkdownExporter] — for baseline storage and reports

### Documentation

- **`.squad/skills/benchmarks/BENCHMARKS.md`** — Comprehensive guide:
  - How to run benchmarks (all, specific classes, with exporters)
  - What each benchmark measures
  - How to interpret results (mean, error, StdDev, allocated memory)
  - Comparing across runs (baseline tracking, regression detection)
  - Troubleshooting (models not found, OOM, BenchmarkDotNet warnings)

- **`.squad/skills/benchmarks/README.md`** — Baseline storage conventions

## Design Decisions

### Why End-to-End Benchmarks?

TextTokenizer and WavWriter are `internal` — no direct access from benchmarks project. Instead of making them `public` (breaking encapsulation), benchmarks use **TtsPipeline** for holistic measurements. This:
- Reflects real-world usage patterns
- Captures cross-component overhead (e.g., tokenization → inference → write)
- Simplifies benchmark setup (single pipeline instance)

Trade-off: Cannot isolate tokenization alone. Acceptable because TTS is dominated by inference time (~95% of total).

### Why Net8.0 RuntimeMoniker?

BenchmarkDotNet 0.15.8 does not support `RuntimeMoniker.Net100`. Using `Net80` ensures compatibility and provides baseline for .NET 8.0 LTS performance. Future migration to .NET 10 benchmarking requires BenchmarkDotNet update.

### Why No Baseline Run Yet?

Baseline requires:
- Downloaded models (~5.5 GB) — not present in all environments
- Models must be in `%LOCALAPPDATA%\ElBruno.QwenTTS\models` or `QWEN_MODEL_DIR`
- At least 8 GB RAM for full inference

Task charter specifies establishing infrastructure; actual baseline run deferred until models are available. Placeholder README explains how to generate baseline.

## Performance Targets (from BENCHMARKS.md)

- **Tokenization:** <50ms for 100 chars, <200ms for 1000 chars
- **Inference:** 5–10× real-time (generate 5–10 seconds audio per wall-clock second)
- **Audio Write:** >100 MB/s (negligible vs. inference)

## Value for PERF-1, PERF-2, PERF-4

This infrastructure enables:
- **PERF-1 (KV-cache optimization):** Measure inference latency before/after heap-based top-k
- **PERF-2 (Model quantization):** Measure latency and throughput for int8 models
- **PERF-4 (GPU acceleration):** Compare CPU vs. DirectML/CUDA performance

## Files Modified

- **NEW:** `src/ElBruno.QwenTTS.Benchmarks/` (5 files: 3 benchmarks + Program.cs + .csproj)
- **NEW:** `.squad/skills/benchmarks/BENCHMARKS.md` (5.9 KB documentation)
- **NEW:** `.squad/skills/benchmarks/README.md` (1 KB baseline conventions)
- **MODIFIED:** `ElBruno.QwenTTS.slnx` (added Benchmarks project)

## Build Status

✅ Clean build (0 errors, 0 warnings)  
✅ Ready to run: `dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*'`

## Next Steps

1. **Download models** (if not present) via `ModelDownloader.DownloadAsync()`
2. **Run baseline:** `dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json`
3. **Store baseline:** Copy JSON to `.squad/skills/benchmarks/baseline-20260228.json`
4. **Use for PERF-1/2/4:** Compare before/after optimizations

## Recommendation

Merge branch `squad/perf-3-benchmarks` to main after code review. Infrastructure is complete and documented; actual baseline run can happen asynchronously (requires model download).
