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

## Voice-cloning workflow controls

The RCL exposes composable controls for a host-owned voice-cloning workflow:

| Component | Purpose |
|-----------|---------|
| `VoiceCloneReferenceAudioInput` | Selects a WAV file or records browser audio, then emits `VoiceCloneReferenceAudio`. |
| `VoiceCloneForm` | Collects synthesis text and an optional reference transcript, validates the request, and emits `VoiceCloneRequest`. |
| `VoiceCloneWorkflowStatus` | Displays host-supplied `VoiceCloneProgress` or an error. |

`VoiceCloneReferenceAudio` contains browser-provided file name, content type, byte content, and acquisition source. `VoiceCloneRequest` contains the synthesis text, reference audio, and optional `ReferenceTranscript`.

The controls do **not** save files, initialize/download models, create service instances, or run inference. Hosts own those operations and wire them through callbacks. A typical host saves `referenceAudio.Content` according to its retention policy, reports its readiness/download state separately, and passes the resulting request to its own serialized inference service.

```razor
<VoiceCloneReferenceAudioInput Value="@referenceAudio"
                               ValueChanged="SaveReferenceAsync"
                               OnError="ShowErrorAsync" />
<VoiceCloneForm Text="@text"
                TextChanged="SetTextAsync"
                ReferenceAudio="@referenceAudio"
                ReferenceTranscript="@transcript"
                ReferenceTranscriptChanged="SetTranscriptAsync"
                IsSubmitting="@isSubmitting"
                IsDisabled="@(!modelIsReady)"
                OnSubmit="CloneAsync"
                OnCancel="CancelAsync" />
<VoiceCloneWorkflowStatus Progress="@progress" Error="@error" />
```

### Reference audio and browser recording

- Uploads are limited to WAV files (`.wav`, `audio/wav`, or `audio/x-wav`); hosts set `MaxFileSize` to their own policy.
- Recording requires `navigator.mediaDevices.getUserMedia` and `AudioContext`. The component reports an explicit unavailable state when those APIs or microphone permission are unavailable.
- The included recorder produces a mono 16-bit PCM WAV. Hosts must include both static assets:

```html
<link rel="stylesheet" href="_content/ElBruno.QwenTTS.BlazorComponents/qwen-tts-components.css" />
<script src="_content/ElBruno.QwenTTS.BlazorComponents/qwen-tts-recording.js"></script>
```

Microphone recording normally requires HTTPS (with localhost treated as a secure context) and a browser with the required APIs.

### Optional ICL transcript and cancellation

An empty `ReferenceTranscript` requests embedding-only cloning. A non-empty transcript is forwarded unchanged so a host can select its ICL-capable inference path. Hosts should surface failures through `Error`, own cancellation token lifetimes, and keep the form disabled while an operation is active. The RCL invokes `OnCancel`; it never assumes how a host can interrupt or clean up inference.
