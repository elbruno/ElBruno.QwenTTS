# Web App

The **QwenTTS.Web** Blazor Server application provides a browser-based UI for generating speech with Qwen TTS.

## Quick Start

```bash
dotnet run --project src/QwenTTS.Web
```

Then open [http://localhost:5153](http://localhost:5153) in your browser.

> **Note:** The Podcast Generator is a separate app — see [Podcast Generator](podcast.md).

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
    "ModelDir": "python/onnx_runtime"
  }
}
```

Relative paths are resolved from the repository root. You can also use an absolute path.

## Architecture

- **Blazor Server** with interactive SSR — all TTS processing runs server-side
- **TtsPipelineService** — singleton service wrapping `TtsPipeline` with thread-safe (semaphore) access
- Generated WAV files are saved to `wwwroot/generated/` and served as static files
- File parsing reuses the same logic as the [File Reader](file-reader.md) CLI app

## Running with a Custom Port

```bash
dotnet run --project src/QwenTTS.Web -- --urls "http://localhost:8080"
```
