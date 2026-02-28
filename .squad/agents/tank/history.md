# Project Context

- **Owner:** Bruno Capuano
- **Project:** Qwen3-TTS → ONNX → C# .NET 10 console app for local voice generation
- **Stack:** Python (PyTorch, transformers, ONNX), C# (.NET 10, ONNX Runtime), Qwen3-TTS
- **Created:** 2026-02-21T15:38Z

## Learnings

- **Build cleanliness:** Successfully validated zero-warning clean build across 8 projects (net8.0 + net10.0 targets). Neo's compiler warning fixes complete and verified.
- **Test coverage:** 38 total tests passing (28 Core + 10 VoiceCloning). xUnit test infrastructure stable and comprehensive. SEC-1 adds 9 validation tests.
- **Multi-target support:** Build succeeds for both .NET 8.0 and .NET 10.0 targets with no platform-specific issues.
- **Input validation testing:** Comprehensive edge case coverage for null, empty, and length boundary conditions. Validation logic is deterministic and testable without requiring full model files.
- **SEC-1 validation:** Neo's input validation implementation is production-ready. Null checks, empty checks, and 10k character limits all correctly implemented with proper exception types and messages.
- **Validation test patterns:** Unit tests that verify validation logic directly (without async/model dependencies) are more reliable than integration tests. Boundary condition testing (n-1, n, n+1) catches off-by-one errors.
- **2026-02-28 SEC-1 validation complete:** Wrote 9 edge case tests (Sec1ValidationTests.cs) covering null, empty, length boundaries, Unicode handling, and validation order. All 38 tests passing (28 Core + 10 VoiceCloning). Confidence: HIGH. Implementation is production-ready.
- **2026-02-28 SEC-3 file size validation complete:** Wrote 11 comprehensive boundary tests (Sec3FileSizeTests.cs) for ONNX (2GB limit) and NPY (500MB limit) file size checks. Tests: just-under, at-boundary (inclusive), just-over, and large overflow cases. All tests use FileStream streaming I/O to avoid memory exhaustion. All 49 tests passing (39 Core + 10 VoiceCloning). Confidence: HIGH. Neo's implementation in NpyReader, LanguageModel, and Vocoder is production-ready.
- **2026-02-28 PERF-3 BenchmarkDotNet setup complete:** Created ElBruno.QwenTTS.Benchmarks project with BenchmarkDotNet 0.15.8. Three benchmark classes: TokenizationBenchmark (text processing 100/1000 char, CJK), InferenceBenchmark (TTS synthesis short/medium/CJK), AudioWriteBenchmark (WAV write 1s/3s/5s). All benchmarks use TtsPipeline end-to-end measurements. Documentation in .squad/skills/benchmarks/BENCHMARKS.md covers setup, running, interpreting results, and baseline comparison. Build clean (0 errors, 0 warnings). Establishes infrastructure for measuring PERF-1, PERF-2, PERF-4 improvements.
- **BenchmarkDotNet patterns:** Separate benchmark project (ElBruno.QwenTTS.Benchmarks) as console app with OutputType=Exe. Use [SimpleJob(RuntimeMoniker.Net80)] for .NET 8.0 target. [MemoryDiagnoser] tracks allocations. [JsonExporter] for baseline storage. GlobalSetup loads models once; GlobalCleanup disposes resources. Since TextTokenizer and WavWriter are internal, benchmarks use TtsPipeline for full end-to-end measurements. Environment variable QWEN_MODEL_DIR allows custom model paths. Release build required for accurate measurements.
- **Benchmark design:** End-to-end benchmarks (tokenization + inference + vocoder + write) provide holistic performance view. Separate short/medium/long texts capture scaling behavior. CJK text benchmarks validate Unicode handling performance. Memory diagnostics (Gen 0/1/2 collections) reveal GC pressure. BenchmarkDotNet's statistical analysis (mean, error, StdDev) provides confidence in measurements. JSON export enables baseline tracking and regression detection.
- **Baseline interpretation:** Mean execution time is primary metric; compare across runs for regression detection (>10% increase = investigate). Memory allocation tracks heap pressure (>20% increase = memory leak). StdDev reveals non-deterministic behavior (GC pauses, disk I/O). Baseline runs require: downloaded models (~5.5 GB), 8+ GB RAM, Release build, minimal background processes. Results vary by hardware; always compare on same machine for consistency.

