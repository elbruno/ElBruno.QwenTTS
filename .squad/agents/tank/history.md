# Project Context

- **Owner:** Bruno Capuano
- **Project:** Qwen3-TTS → ONNX → C# .NET 10 console app for local voice generation
- **Stack:** Python (PyTorch, transformers, ONNX), C# (.NET 10, ONNX Runtime), Qwen3-TTS
- **Created:** 2026-02-21T15:38Z

## Learnings

- **Build cleanliness:** Successfully validated zero-warning clean build across 8 projects (net8.0 + net10.0 targets). Neo's compiler warning fixes complete and verified.
- **Test coverage:** 29 total tests passing (19 Core + 10 VoiceCloning). xUnit test infrastructure stable and comprehensive.
- **Multi-target support:** Build succeeds for both .NET 8.0 and .NET 10.0 targets with no platform-specific issues.
- **No edge cases found:** Full dependency chain builds cleanly; no missing references or version conflicts detected.
