# Blazor Components Package

`ElBruno.QwenTTS.BlazorComponents` is a Razor Class Library that provides reusable UI components for Qwen TTS and voice-cloning flows.

## Install

```bash
dotnet add package ElBruno.QwenTTS.BlazorComponents
```

## Register services

```csharp
using ElBruno.QwenTTS.BlazorComponents.Extensions;

builder.Services.AddQwenTtsBlazorComponents();
```

## Components

### `TtsInputPanel`
- Parameters: `TtsService`, `VoicePreset`, `OnAudioReady`
- Renders text input + synthesize action and emits generated audio stream.

### `VoiceSamplePlayer`
- Parameters: `AudioStream`, `Label`, `AutoPlay`
- Plays synthesized or reference audio.

### `VoiceCloningSamplePicker`
- Parameters: `VoiceCloningService`, `OnVoiceReady`
- Accepts uploaded reference audio and emits generated `VoiceEmbedding`.

### `ModelDownloadStatus`
- Parameters: `ModelId`, `OnModelReady`
- Displays model availability/download status and initialization progress.

### `SynthesisProgressBar`
- Parameters: `TtsService`, `ShowChunkCount`
- Displays synthesis progress UI.
