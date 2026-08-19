# Voice Cloning with Qwen3-TTS Base Model

Clone any voice from a 3-second audio sample using the Qwen3-TTS Base model with ECAPA-TDNN speaker encoder.

## Overview

The `ElBruno.QwenTTS.VoiceCloning` NuGet package extends the core TTS library with voice cloning capabilities. It uses:

- **ECAPA-TDNN Speaker Encoder** — extracts a 1024-dim speaker embedding from reference audio
- **Qwen3-TTS Base Model** — generates speech conditioned on the speaker embedding
- **Mel-spectrogram DSP** — preprocesses audio for the speaker encoder (24 kHz, 128 mel bins)

## Installation

```bash
dotnet add package ElBruno.QwenTTS.VoiceCloning
```

This automatically includes `ElBruno.QwenTTS` (the core library) as a dependency.

## Quick Start

```csharp
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

// Create pipeline (auto-downloads ~5.5 GB Base model on first run)
var cloner = await VoiceClonePipeline.CreateAsync();

// Clone a voice from reference audio
await cloner.SynthesizeAsync(
    text: "Hello, this is my cloned voice!",
    referenceAudioPath: "reference_speaker.wav",
    outputPath: "cloned_output.wav",
    language: "english"
);
```

## Advanced Usage

### Reuse Speaker Embeddings

For multiple utterances with the same cloned voice, extract the embedding once:

```csharp
var cloner = await VoiceClonePipeline.CreateAsync();

// Extract embedding once
var embedding = cloner.ExtractSpeakerEmbedding("reference_speaker.wav");

// Synthesize multiple utterances with the same voice
await cloner.SynthesizeWithEmbeddingAsync("First sentence.", embedding, "out1.wav", "english");
await cloner.SynthesizeWithEmbeddingAsync("Second sentence.", embedding, "out2.wav", "english");
await cloner.SynthesizeWithEmbeddingAsync("Third sentence.", embedding, "out3.wav", "english");
```

### Custom Model Directory

```csharp
var cloner = await VoiceClonePipeline.CreateAsync(
    modelDir: @"C:\my\custom\model\path"
);
```

## Reference Audio Tips

For best voice cloning results:
- Use **3+ seconds** of clear speech
- **24 kHz mono WAV** is preferred (other formats are auto-converted)
- Minimize background noise
- Avoid music or overlapping speakers
- The speaker encoder is robust to recording quality, but cleaner audio = better results

## Model Details

| Component | File | Size | Purpose |
|-----------|------|------|---------|
| Speaker Encoder | `speaker_encoder.onnx` | ~34 MB | ECAPA-TDNN voice embedding |
| Talker Prefill | `talker_prefill.onnx` | ~1.7 GB | Language model (full sequence) |
| Talker Decode | `talker_decode.onnx` | ~1.7 GB | Language model (autoregressive) |
| Code Predictor | `code_predictor.onnx` | ~440 MB | Codebook generation |
| Vocoder | `vocoder.onnx` | ~2.7 MB | Audio codes → waveform |
| Embeddings | `embeddings/*.npy` | ~1.4 GB | Lookup tables |

**Total download:** ~5.5 GB (stored at `%LOCALAPPDATA%\ElBruno\QwenTTS-Base`)

## Differences from Standard TTS

| Feature | `ElBruno.QwenTTS` | `ElBruno.QwenTTS.VoiceCloning` |
|---------|-------------------|-------------------------------|
| Model | CustomVoice (9 preset speakers) | Base (any voice) |
| Input | Text + speaker name | Text + reference audio |
| Speaker selection | `"ryan"`, `"emily"`, etc. | Any WAV file |
| Extra dependency | None | Speaker encoder + mel DSP |

## Python Export Scripts

The ONNX models were exported using the scripts in `python/`:

```bash
# Download Base model weights
python download_models.py --model base

# Export speaker encoder
python export_speaker_encoder.py --model-dir models/Qwen3-TTS-0.6B-Base --output-dir onnx_base/

# Export LM (talker + code predictor)
python export_lm.py --model-dir models/Qwen3-TTS-0.6B-Base --output-dir onnx_base/

# Export embeddings
python export_embeddings.py --model-dir models/Qwen3-TTS-0.6B-Base --output-dir onnx_base/embeddings
```

## Web UI

The Blazor web app includes a **Voice Clone** page built from reusable RCL controls:

```bash
dotnet run --project src/ElBruno.QwenTTS.Web
```

Open [http://localhost:5153/voice-clone](http://localhost:5153/voice-clone) to:

1. **Record** browser audio as a mono 16-bit PCM WAV, or upload a WAV file
2. **Preview and download** the host-saved reference audio
3. Optionally enter its transcript to enable ICL voice cloning
4. **Type text**, submit the request, and download the generated WAV

The controls do not perform filesystem work, model setup, or inference. The Web host owns reference persistence, Base-model readiness/download, serialized inference, cancellation, and generated-file retention. Browser recording requires microphone permission, `getUserMedia`, and `AudioContext`; the UI explicitly reports unsupported browsers or unavailable APIs. The model-free callback-flow demo is available at [http://localhost:5153/voice-clone-demo](http://localhost:5153/voice-clone-demo).

See [Web App](web-app.md) and [Blazor Components](blazor-components.md) for host integration details.

## Aspire Sample: Record-and-Clone

`samples/BlazorQwenTtsDemo` (orchestrated by `samples/BlazorQwenTtsDemo.AppHost`) includes the same record-and-clone flow at `/voice-clone`, built on its own `VoiceClonePipelineService` under `samples/BlazorQwenTtsDemo/Services`. It follows the identical host-owns-persistence pattern as the Web app, reuses the sample's existing `VoiceCloningModelDownloadController` for readiness gating, and produces the same real `voice_clone.*` GenAI OpenTelemetry traces (from `VoiceClonePipeline`) — visible in the Aspire dashboard when the sample runs via `aspire start`. See [BlazorQwenTtsDemo Sample App](blazor-qwen-tts-demo.md) for details.
