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

