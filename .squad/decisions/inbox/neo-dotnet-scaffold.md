# Decision: .NET project scaffold and package choices

**Date:** 2026-02-21
**By:** Neo (.NET Developer)
**Status:** Accepted

## Context
Needed to establish the C# project structure for Qwen3-TTS ONNX inference.

## Decision
- Target **net10.0** (SDK 10.0.102 confirmed available)
- Use **Microsoft.ML.OnnxRuntime** 1.24.2 for ONNX inference
- Use **Microsoft.ML.Tokenizers** 2.0.0 for BPE tokenization (avoids hand-rolling tokenizer)
- Use **NAudio** 2.2.1 as available audio library (WavWriter is pure .NET for now — NAudio may be used for playback later)
- Simple `string[] args` parsing for CLI (no System.CommandLine dependency — keeps it lightweight)
- WavWriter outputs 16-bit PCM WAV at 24 kHz (Qwen3-TTS default sample rate)
- Project structure: Models / Pipeline / Audio separation for clean layering

## Consequences
- All model classes throw `NotImplementedException` until ONNX models are exported by the Python side
- Prompt format (`<speaker>`, `<instruct>` tags) is a placeholder — needs validation against actual model expectations
