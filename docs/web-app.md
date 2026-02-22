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

## Voice Clone Page

The app includes a dedicated **Voice Clone** page at `/voice-clone` that lets you clone any voice from a short audio sample.

### How to use

1. Navigate to **🎭 Voice Clone** in the top navigation bar
2. **Record** your voice using the microphone button (3+ seconds recommended), or **Upload** a WAV file
3. Preview and download the recorded/uploaded reference audio
4. Type the text you want to synthesize with the cloned voice
5. Select a language and click **🎭 Generate Cloned Speech**

### Recording details

- The microphone recorder captures audio in your browser and automatically converts it to **24 kHz mono 16-bit PCM WAV** — the exact format needed by the speaker encoder
- A live timer shows recording duration; aim for **3+ seconds** of clear speech
- You can download the recorded WAV file for reuse later

### Backend

- Uses `VoiceClonePipelineService` (singleton, thread-safe) wrapping `VoiceClonePipeline`
- The **Base model** (~5.5 GB) downloads automatically on first visit — this is a separate model from the CustomVoice model used on the main TTS page
- Speaker embedding is extracted once per reference audio and cached for multiple synthesis calls
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
- Client-side JavaScript (`js/audioRecorder.js`) handles microphone recording and WAV conversion
- File parsing reuses the same logic as the [File Reader](file-reader.md) CLI app

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Generate Speech | `/` | Text/file input → preset voice selection → audio generation |
| Voice Clone | `/voice-clone` | Record/upload reference audio → text input → cloned voice generation |

## Running with a Custom Port

```bash
dotnet run --project src/ElBruno.QwenTTS.Web -- --urls "http://localhost:8080"
```
