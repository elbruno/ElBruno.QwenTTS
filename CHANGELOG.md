# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v1.4.0] - 2026-07-25

### Added

- **ICL (In-Context Learning) mode for voice cloning** ([#32](https://github.com/elbruno/ElBruno.QwenTTS/issues/32))
  - New `SynthesizeAsync` overload accepting `refText` parameter for higher-quality voice cloning
  - When reference text transcript is provided alongside reference audio, the model preserves accent characteristics (e.g., Trump voice speaking Chinese retains English accent)
  - New `SpeechTokenizer` class wrapping `tokenizer12hz_encode.onnx` for audio → codec code extraction
  - New `GenerateWithSpeakerEmbeddingAndRefText` method in `LanguageModel` for ICL prefill embedding
  - Speech tokenizer encoder ONNX export script (`python/export_speech_tokenizer.py`)
  - `tokenizer12hz_encode.onnx` added to `VoiceCloningDownloader` expected files
  - 4 new ICL-specific unit tests

### Fixed

- SpeechTokenizer tensor shape/name compatibility with ONNX export (input: 3D `[B,1,T]`, output transpose `[B,16,T]→[B,T,16]`)

## [v1.3.0] - 2026-07-24

### Added

- **1.7B model quality and performance improvements** ([#30](https://github.com/elbruno/ElBruno.QwenTTS/issues/30))

## [v1.2.3] - 2026-04-06

First stable release (previously v1.2.3-preview).

### Fixed

- **CP projection span dimension mismatch for 1.7B models** ([#29](https://github.com/elbruno/ElBruno.QwenTTS/issues/29))
  - Span constructors in `LanguageModel.cs` now consistently use `cpInputDim` instead of `_cpHiddenSize`
  - Added `Debug.Assert` guards at dimension boundaries (zero-cost in Release builds)
  - Added CP projection input/output dimension validation in `EmbeddingStore.cs`

### Added

- Extracted `BuildCpPrefillDirect` and `AccumulateCpEmbedding` as testable internal static methods
- 21 new parameterized dimension tests covering 0.6B, 1.7B, and mismatched-dimension configurations

## [1.2.2-preview] - 2026-04-05

### Fixed

- **Squad Protected Branch Guard** — now allows essential config files (team.md, routing.md, ceremonies.md) on main without triggering protection rule violations
- **GitHub Actions workflows** — configured all Squad release/preview/insider workflows with proper .NET build/test commands
- **squad-promote.yml** — now uses .NET csproj version instead of package.json
- **Node.js 24 compatibility** — added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` environment variable to all 13 workflows
- **Git tracking** — removed .squad/ runtime state files from main branch

## [1.2.1-preview] - 2026-04-05

### Fixed

- **CP projection bias dimension mismatch for 1.7B model variant** ([#28](https://github.com/elbruno/ElBruno.QwenTTS/issues/28))
  - Added `hidden_size` to `CodePredictorConfig` to read `code_predictor.hidden_size` from config.json
  - Fixed `EmbeddingStore.CpProjection()` bias loop to use projection weight output dimension instead of `_cpHiddenSize`
  - Fixed `LanguageModel` CP input dimension to use config-driven value via `CpModelHiddenSize`
  - Added load-time validation for projection weight/bias shape consistency

### Added

- 16 regression tests for the CP dimension mismatch scenario

## [1.2.0] - 2025-07-25

### Fixed

- **1.7B model text truncation** — fixed Code Predictor input dimension mismatch that caused only first ~2 words to be generated ([#27](https://github.com/elbruno/ElBruno.QwenTTS/issues/27))
  - Removed `small_to_mtp_projection` from ONNX graph; projection now applied externally in C#
  - Re-exported `code_predictor.onnx` for 1.7B with correct input shape (1024-dim)
  - `EmbeddingStore` loads optional `cp_projection_weight.npy` / `cp_projection_bias.npy` and applies projection during CP prefill
  - Full backward compatibility: old models without projection files continue to work

### Added

- 41 new tests covering CP projection math, input dimension contracts for 0.6B/1.7B, and model downloader variant file lists
- `ModelDownloader` variant-aware file lists with CP projection NPY files for 1.7B

## [1.1.1] - 2025-07-24

### Fixed

- **NPY file size limit raised to 2 GB** — unblocks 1.7B model's `text_embedding.npy` (~1.2 GB) ([#25](https://github.com/elbruno/ElBruno.QwenTTS/issues/25))
- **Vocoder ONNX size limit raised to 8 GB** — consistent with LanguageModel limits for large model support

### Added

- 10 new NpyReader size-limit integration tests exercising real validation paths

## [1.1.0] - 2026-04-02

### Added

- **Qwen3-TTS 1.7B model support** — run either 0.6B or 1.7B model variant ([#26](https://github.com/elbruno/ElBruno.QwenTTS/issues/26))
  - `QwenModelVariant` enum (`Qwen06B`, `Qwen17B`) drives model selection
  - `QwenModelVariantConfig` maps each variant to HuggingFace repo, model directory, and dimension defaults
  - Config-driven dimensions: hidden_size, num_layers, head_dim read from model's `config.json` at runtime
  - Zero hardcoded dimension constants — `LanguageModel` and `EmbeddingStore` fully parameterized
  - `TtsPipeline.CreateAsync()` accepts optional `variant` parameter (defaults to 0.6B for backward compatibility)
  - Variant-specific model download directories under `DefaultModelDir`
- **Instruct control API** — natural-language style control for 1.7B models
  - `SynthesizeAsync` accepts optional `instruct` parameter (e.g., "speak with excitement")
  - Instruct gated by variant: 1.7B flows through to prompt, 0.6B warns and ignores
  - `QwenModelVariantConfig.SupportsInstruct()` as single source of truth
- **CLI `--variant` and `--instruct` flags** — select model variant and instruct text from command line
- **FileReader `--variant` flag** — batch process with either model variant
- **Blazor Web app variant support** — variant selector dropdown, dynamic instruct field (disabled for 0.6B)
- **47 new unit tests** — `ModelVariantTests`, `ModelVariantDownloaderTests`, `TtsPipelineVariantTests`
- **Python export updates** — `read_model_dims(config)` replaces hardcoded constants; 1.7B download support

- **`ITextToSpeechClient`** — MEAI-aligned service contract for text-to-speech with `SynthesizeToMemoryAsync` and `SynthesizeStreamingAsync`
- **`QwenTextToSpeechClient`** — production-ready implementation with:
  - Thread-safe lazy initialization via `SemaphoreSlim` (model loads on first use, shared across concurrent callers)
  - `SynthesizeToMemoryAsync` — returns audio as `byte[]` with automatic temp file cleanup
  - `SynthesizeStreamingAsync` — `IAsyncEnumerable<TextToSpeechStreamingUpdate>` with session lifecycle events (Open → AudioChunk → Close)
  - Proper `IDisposable` with cleanup of ONNX sessions, semaphore, and temp files
  - GPU support via `sessionOptionsFactory` / `vocoderSessionOptionsFactory` parameters
- **`TextToSpeechResponse`** — response type with `AudioData`, `MediaType`, `SampleRate`, `ModelId`
- **`TextToSpeechStreamingUpdate`** — streaming update with `Kind` enum (`SessionOpen`, `AudioChunk`, `SessionClose`)
- **`TextToSpeechOptions`** — request options with `VoiceId`, `Language`, `Instruct`, `ModelId`
- **`AddQwenTextToSpeechClient()`** — DI extension method for registering `ITextToSpeechClient` as singleton
- **Related Projects** section in README with link to [ElBruno.PersonaPlex](https://github.com/elbruno/ElBruno.PersonaPlex)
- **GPU Acceleration** — configurable `SessionOptions` injection for CUDA and DirectML support
  - `TtsPipeline` and `VoiceClonePipeline` accept optional `Func<SessionOptions>?` parameter
  - `OrtSessionHelper` static class with `CreateCpuOptions()`, `CreateCudaOptions()`, `CreateDirectMlOptions()`
  - All ONNX sessions (LanguageModel, Vocoder, SpeakerEncoder) respect the configured execution provider
  - Default behavior unchanged (CPU with full graph optimization)
  - [GPU Acceleration docs](docs/gpu-acceleration.md)

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
