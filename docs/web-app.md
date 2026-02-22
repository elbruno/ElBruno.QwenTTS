# Web App

The **ElBruno.QwenTTS.Web** Blazor Server application provides a browser-based UI for generating speech with Qwen TTS.

## Quick Start

```bash
dotnet run --project src/ElBruno.QwenTTS.Web
```

Then open [http://localhost:5153](http://localhost:5153) in your browser.

> **Note:** The Podcast Generator has moved to its own repository: [ElBruno.Podcast.TTS](https://github.com/elbruno/ElBruno.Podcast.TTS)

## Features

- **Type text** or **upload a file** (.txt, .srt, .md) for speech generation
- **Speaker selection** — choose from all available voices (Ryan, Serena, Vivian, Aiden, etc.)
- **Language selection** — English, Spanish, Chinese, Japanese, Korean, or auto-detect
- **Voice instructions** — optional style prompts (e.g., "speak slowly and calmly")
- **Audio playback** — listen to generated audio directly in the browser
- **Download** — save generated WAV files locally
- **Batch processing** — uploaded files are split into segments, each generating a separate audio clip

## Configuration

The model directory is configured in `appsettings.json`:

```json
{
  "TTS": {
    "ModelDir": "models"
  }
}
```

Models are downloaded automatically on first request if not already present. You can also use an absolute path to a pre-downloaded model directory.

## Architecture

- **Blazor Server** with interactive SSR — all TTS processing runs server-side
- **TtsPipelineService** — singleton service wrapping `TtsPipeline` with thread-safe (semaphore) access
- Generated WAV files are saved to `wwwroot/generated/` and served as static files
- File parsing reuses the same logic as the [File Reader](file-reader.md) CLI app

## Running with a Custom Port

```bash
dotnet run --project src/ElBruno.QwenTTS.Web -- --urls "http://localhost:8080"
```
