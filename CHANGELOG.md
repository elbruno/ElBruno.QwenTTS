# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-22

### Added

- **ElBruno.QwenTTS** NuGet package — complete Qwen3-TTS inference pipeline for .NET
- `TtsPipeline` — full TTS orchestrator: text → tokenize → LM → vocoder → WAV
- `TtsPipeline.CreateAsync()` — factory method with automatic model download from HuggingFace
- `ModelDownloader` — auto-download 33 model files (~5.5 GB) with byte-level progress reporting
- `ModelDownloader.DefaultModelDir` — shared model directory (`%LOCALAPPDATA%/ElBruno.QwenTTS/models`)
- `ModelDownloader.IsModelDownloaded` — check if models are ready without downloading
- Multi-speaker support: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee
- Multi-language support: english, spanish, chinese, japanese, korean
- Instruction-based voice control (e.g., "speak with excitement")
- 24 kHz mono WAV output
- CLI app (`ElBruno.QwenTTS`)
- File reader app (`ElBruno.QwenTTS.FileReader`) for batch text/SRT → audio
- Web app (`ElBruno.QwenTTS.Web`) — Blazor UI for speech generation
- Unit tests (`ElBruno.QwenTTS.Core.Tests`)
