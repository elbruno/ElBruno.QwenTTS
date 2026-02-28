# Baseline Performance Results

This directory stores baseline performance benchmark results for tracking performance regressions and improvements over time.

## Baseline Format

Baseline files are JSON exports from BenchmarkDotNet:

```
baseline-YYYYMMDD.json
```

## Generating Baseline

```bash
dotnet run --project src/ElBruno.QwenTTS.Benchmarks -c Release -- -f '*' --exporters json
```

Then copy the full JSON report to this directory:

```bash
# Windows
copy src\ElBruno.QwenTTS.Benchmarks\BenchmarkDotNet.Artifacts\results\*-report-full.json .squad\skills\benchmarks\baseline-20260228.json

# Linux/macOS
cp src/ElBruno.QwenTTS.Benchmarks/BenchmarkDotNet.Artifacts/results/*-report-full.json .squad/skills/benchmarks/baseline-20260228.json
```

## Note

Baseline runs require:
- Downloaded models (~5.5 GB)
- At least 8 GB RAM
- Release build configuration
- Minimal background processes

Baseline results will vary based on hardware. For consistent comparisons, always run on the same machine configuration.
