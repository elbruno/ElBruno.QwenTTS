# Copilot Instructions

This is a **Squad-managed** repository (Squad v0.5.2) — an AI team framework where a Coordinator orchestrates specialist agents.

## Application

This repo contains **ElBruno.QwenTTS** — a C# .NET library and tools for running Qwen3-TTS text-to-speech locally using ONNX Runtime. The Core library is published to NuGet as `ElBruno.QwenTTS`. Pre-exported ONNX models are hosted on HuggingFace at `elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX`.

## Build & Test

```bash
dotnet build          # Build all 5 projects
dotnet test           # Run xUnit tests (ElBruno.QwenTTS.Core.Tests)
dotnet pack src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj -c Release -o artifacts  # Create NuGet package
```

## Project Structure

```
ElBruno.QwenTTS.slnx                    # Solution file (5 projects)
src/ElBruno.QwenTTS.Core/               # Core library (NuGet: ElBruno.QwenTTS, targets net8.0+net10.0)
  Pipeline/TtsPipeline.cs               # Full TTS orchestrator + CreateAsync factory
  Pipeline/ModelDownloader.cs            # Auto-download models from HuggingFace (~5.5 GB)
  Models/                               # TextTokenizer, LanguageModel, Vocoder, EmbeddingStore
  Audio/WavWriter.cs                    # WAV file writer (24 kHz, 16-bit PCM)
src/ElBruno.QwenTTS/                    # CLI console application
src/ElBruno.QwenTTS.FileReader/         # Batch file reader (text/SRT → audio)
src/ElBruno.QwenTTS.Web/               # Blazor web app — text-to-speech UI (port 5153)
src/ElBruno.QwenTTS.Core.Tests/        # xUnit unit tests
docs/                                   # All documentation
python/                                 # ONNX export & HuggingFace download tools (optional)
images/nuget_01.png                     # NuGet package icon (512x512)
```

## Key Technical Details

- **Shared model directory**: `ModelDownloader.DefaultModelDir` → `%LOCALAPPDATA%/ElBruno.QwenTTS/models` (Windows), `~/.local/share/ElBruno.QwenTTS/models` (Linux/macOS). All apps share this.
- **TtsPipeline.CreateAsync()**: Factory method that auto-downloads models if missing, with byte-level progress reporting.
- **ModelDownloadProgress**: Record with `CurrentFile`, `TotalFiles`, `FileName`, `BytesDownloaded`, `TotalBytes`, percentage properties.
- **33 model files**: 7 ONNX (+.data), 15 CP codec embeddings, 8 other embeddings/config, 2 tokenizer files, 1 speaker_ids.json.
- **HuggingFace URL pattern**: `https://huggingface.co/{repoId}/resolve/main/{filename}` — no API token needed for public repos.
- **Web app async init**: `TtsPipelineService` has `InitializeAsync()` called from page `OnInitializedAsync()`. Uses `IDisposable` to unsubscribe from events.
- **TTS inference offloaded**: `TtsPipelineService.GenerateAsync` wraps `SynthesizeAsync` in `Task.Run()` to keep Blazor UI responsive.

## NuGet Publishing

- **Package name**: `ElBruno.QwenTTS` (PackageId in Core csproj, distinct from CLI project name)
- **Workflow**: `.github/workflows/publish.yml` — triggered on GitHub release or manual dispatch
- **Authentication**: OIDC via `NuGet/login@v1`, requires `NUGET_USER` secret in `release` environment
- **Version**: Determined from release tag (`v1.0.0` → `1.0.0`), manual input, or csproj fallback

## Documentation Convention

- **Only `README.md`, `LICENSE`, and `CHANGELOG.md` live at the repo root.** All other documentation goes in `docs/`.
- Documentation files use kebab-case naming (e.g., `docs/getting-started.md`, `docs/core-library.md`).
- The main `README.md` should be concise with quick start commands and links to `docs/` for details.

## Key Conventions

### File Ownership Model

Squad distinguishes **user-owned** files (`.squad/`) from **Squad-owned** files (`.squad-templates/`, `.github/agents/squad.agent.md`). Init skips existing user files (idempotent); upgrades only overwrite Squad-owned files.

### Git Merge Strategy

`.gitattributes` uses `merge=union` for append-only files (decisions, history, logs, orchestration-log). This enables conflict-free merging of squad state across branches.

### Team File Structure

`team.md` **must** have a section titled exactly `## Members` — GitHub workflows (`squad-heartbeat.yml`, `squad-issue-assign.yml`, `squad-triage.yml`, `sync-squad-labels.yml`) hardcode this header for label automation.

### Branch Naming

Squad branches follow: `squad/{issue-number}-{kebab-case-slug}` (e.g., `squad/42-fix-login-validation`).
