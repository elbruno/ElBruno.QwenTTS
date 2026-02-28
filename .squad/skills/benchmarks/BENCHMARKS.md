# Qwen TTS Performance Benchmarks

This document describes the BenchmarkDotNet performance benchmarking infrastructure for ElBruno.QwenTTS.

## Overview

Performance benchmarks measure three critical hot paths in the TTS pipeline:

1. **TokenizationBenchmark** — End-to-end text processing (tokenization + inference + audio write) for various input sizes
2. **InferenceBenchmark** — Full TTS synthesis across different text lengths and languages
3. **AudioWriteBenchmark** — WAV file writing performance for various audio durations

## Prerequisites

- Models must be downloaded before running benchmarks
- Set `QWEN_MODEL_DIR` environment variable if models are not in the default location (`%LOCALAPPDATA%\ElBruno.QwenTTS\models`)
- Release build configuration for accurate performance measurements

## Running Benchmarks

### Run All Benchmarks

```bash
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*'
```

### Run Specific Benchmark Class

```bash
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*TokenizationBenchmark*'
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*InferenceBenchmark*'
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*AudioWriteBenchmark*'
```

### Export Results

```bash
# JSON export (for baseline comparison)
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json

# Multiple formats
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json markdown html
```

Results are saved to `src/ElBruno.QwenTTS.Benchmarks/BenchmarkDotNet.Artifacts/results/`.

## Benchmark Details

### TokenizationBenchmark

Measures end-to-end processing performance for different text inputs:

- **Process English 100 chars** — Short English text (~100 characters)
- **Process English 1000 chars** — Long English text (~1000 characters, 10× repeated)
- **Process CJK 100+ chars** — Chinese text with complex Unicode characters

**What it measures:**
- Tokenization overhead (BPE encoding)
- Language model inference time
- Vocoder decoding
- WAV file writing

**Key metrics:**
- Mean execution time (ms)
- Memory allocation (MB)
- Throughput (operations/sec)

### InferenceBenchmark

Measures full TTS synthesis performance:

- **TTS Short (10 words)** — Single sentence (~10 words)
- **TTS Medium (30 words)** — Multi-sentence text (~30 words)
- **TTS CJK (short)** — Chinese text synthesis

**What it measures:**
- Language model autoregressive generation
- KV-cache management efficiency
- Code predictor invocations (31 per generation step)
- Vocoder throughput

**Key metrics:**
- Mean latency (ms) — Time to first audio frame
- Throughput (audio seconds generated per wall-clock second)
- Memory usage (peak allocation)

### AudioWriteBenchmark

Measures WAV file I/O performance:

- **Write short audio (~1s)** — Short sentence
- **Write medium audio (~3s)** — Medium sentence
- **Write long audio (~5s)** — Long sentence

**What it measures:**
- File system I/O overhead
- PCM float32 → int16 conversion
- WAV header encoding

**Key metrics:**
- Write throughput (MB/s)
- Latency (ms)
- GC pressure (Gen 0/1/2 collections)

## Interpreting Results

BenchmarkDotNet outputs:

| Method | Mean | Error | StdDev | Allocated |
|--------|------|-------|--------|-----------|
| TokenizeEnglishShort | 125.3 ms | 2.4 ms | 2.2 ms | 45 MB |

### Key Columns

- **Mean** — Average execution time across multiple iterations
- **Error** — Half of 99.9% confidence interval
- **StdDev** — Standard deviation of all measurements
- **Allocated** — Total memory allocated (heap allocations)

### Performance Targets

Based on Qwen3-TTS architecture:

- **Tokenization**: <50ms for 100 chars, <200ms for 1000 chars
- **Inference**: 5–10× real-time (generate 5–10 seconds of audio per second of wall time)
- **Audio Write**: >100 MB/s throughput (negligible overhead vs. inference)

## Comparing Results Across Runs

### Establish Baseline

```bash
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json
cp src/ElBruno.QwenTTS.Benchmarks/BenchmarkDotNet.Artifacts/results/*-report-full.json .squad/skills/benchmarks/baseline-$(date +%Y%m%d).json
```

### Compare After Optimization

```bash
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json
# Compare JSON files manually or use BenchmarkDotNet's comparison tools
```

### Regression Detection

- **Mean time increases >10%** → Performance regression (investigate)
- **Memory allocation increases >20%** → Memory leak or inefficient allocation pattern
- **StdDev increases significantly** → Non-deterministic behavior (check GC pauses, disk I/O)

## Continuous Integration

Benchmarks are **not** run in CI by default (too time-consuming). Run manually when:

- Implementing PERF-1 (KV-cache optimization)
- Implementing PERF-2 (model quantization)
- Implementing PERF-4 (GPU acceleration)
- Before merging large refactors

## Troubleshooting

### Models Not Found

**Error:** `Model directory not found: <path>`

**Solution:**
- Run `ModelDownloader.DownloadAsync()` from any project
- Or set `QWEN_MODEL_DIR` environment variable to existing model directory

### Out of Memory

**Solution:**
- Close other applications
- Run benchmarks individually (not all at once)
- Increase system page file size

### BenchmarkDotNet Warnings

BenchmarkDotNet may warn about:
- Precision issues (ignore for initial baseline)
- High outliers (check for background processes)
- GC pressure (expected for ML workloads)

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [Issue #22 — PERF-3 Task Definition](https://github.com/elbruno/ElBruno.QwenTTS/issues/22)
