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
- **Language selection** — English, Spanish, Chinese, Japanese, Korean, Russian, German, French, Portuguese, Italian, or auto-detect
- **Voice instructions** — optional style prompts (e.g., "speak slowly and calmly")
- **Audio playback** — listen to generated audio directly in the browser
- **Download** — save generated WAV files locally
- **Batch processing** — uploaded files are split into segments, each generating a separate audio clip

## Voice Clone Page

The app includes a dedicated **Voice Clone** page at `/voice-clone` that lets you clone any voice from a short audio sample.

### How to use

1. Navigate to **🎭 Voice Clone** in the top navigation bar
2. Choose a WAV reference file or record clear speech in the browser (3+ seconds recommended)
3. Optionally enter the transcript of the reference audio to enable ICL mode
4. Submit the text to synthesize and preview/download the generated WAV

### Recording details

- Upload accepts WAV files only. The RCL validates the file extension and WAV MIME types before invoking the host callback.
- Browser recording requires microphone permission plus `getUserMedia` and `AudioContext`; unavailable APIs produce a visible message instead of silently failing. HTTPS is generally required, except for localhost.
- The bundled recorder creates mono 16-bit PCM WAV. Use 3+ seconds of clear speech for best results.
- The host saves every accepted reference WAV and exposes it for playback/download at `/references/`.

### ICL, readiness, errors, and cancellation

- Leaving the reference transcript empty uses speaker-embedding-only cloning.
- Providing the transcript uses the ICL path, which also requires the speech tokenizer included with the voice-cloning model.
- The host owns Base-model initialization/download state and blocks submission until the model is ready. It saves reference data, serializes inference, and writes generated WAVs to `/generated/`.
- Errors from recording, file persistence, model initialization, and inference are displayed in the page workflow status.
- Cancel requests are forwarded to the host cancellation token. Cancellation can stop queued work and checked stages; an already-running synchronous ONNX inference call completes before the page can show the cancelled state.

### Model-free demo

`/voice-clone-demo` exercises the same reusable controls with deterministic local WAV output. It does not initialize/download models, access microphone hardware unless the user chooses recording, or call external services. Use it to validate the component callback flow without model artifacts.

### Backend

- Uses `VoiceClonePipelineService` (singleton, thread-safe) wrapping `VoiceClonePipeline`
- The **Base model** (~5.5 GB) downloads when the user requests initialization — this is a separate model from the CustomVoice model used on the main TTS page
- The service persists reference WAV files before inference and serializes generation through its semaphore
- Reference audio files are saved to `wwwroot/references/`

## Configuration

The model directory is configured in `appsettings.json`:

```json
{
  "TTS": {
    "ModelDir": "models"
  },
  "VoiceClone": {
    "ModelDir": "models-base"
  }
}
```

Models are downloaded automatically on first request if not already present. You can also use an absolute path to a pre-downloaded model directory. The TTS and Voice Clone pages use different models (CustomVoice and Base, respectively).

## Architecture

- **Blazor Server** with interactive SSR — all TTS processing runs server-side
- **TtsPipelineService** — singleton wrapping `TtsPipeline` with thread-safe (semaphore) access
- **VoiceClonePipelineService** — singleton wrapping `VoiceClonePipeline` for the Voice Clone page
- Generated WAV files are saved to `wwwroot/generated/` and served as static files
- Reference audio files are saved to `wwwroot/references/`
- RCL static assets provide WAV recording: `_content/ElBruno.QwenTTS.BlazorComponents/qwen-tts-recording.js`
- File parsing reuses the same logic as the [File Reader](file-reader.md) CLI app

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Generate Speech | `/` | Text/file input → preset voice selection → audio generation |
| Voice Clone | `/voice-clone` | RCL WAV reference input → optional ICL transcript → Base-model cloning |
| Voice Clone Demo | `/voice-clone-demo` | Model-free deterministic callback-flow demo |

## Running with a Custom Port

```bash
dotnet run --project src/ElBruno.QwenTTS.Web -- --urls "http://localhost:8080"
```
