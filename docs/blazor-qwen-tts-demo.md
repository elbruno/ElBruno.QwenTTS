# BlazorQwenTtsDemo Sample App

The sample app at `samples/BlazorQwenTtsDemo` demonstrates all components from `ElBruno.QwenTTS.BlazorComponents`.

## Run

```bash
dotnet run --project samples/BlazorQwenTtsDemo
```

Open the local URL printed by ASP.NET Core and use the home page to test:

1. Separate `ModelDownloadStatus` controls for CustomVoice TTS and Base voice-cloning models. Each reads the host's shared model cache and downloads only after its own **Download Model** button is selected.
2. `TtsInputPanel` for text input and synthesis trigger.
3. `SynthesisProgressBar` for progress feedback.
4. `VoiceCloningSamplePicker` for reference sample selection/upload.
5. `VoiceSamplePlayer` for generated/reference audio playback.

## Record and clone your voice

The `/voice-clone` page delivers a real, end-to-end record-and-clone scenario:

1. Confirm the **Voice-cloning models** are downloaded (reuses the same `VoiceCloningModelDownloadController`/`VoiceCloningModelDownloadAdapter` shown on the home page — no separate download mechanism).
2. Record a short reference clip directly from your browser microphone (or upload a WAV file, up to 1 MiB) using `VoiceCloneReferenceAudioInput`. The recording is saved to `wwwroot/references` on the host as soon as it is captured.
3. Enter the text to synthesize and, optionally, a transcript of the reference audio to enable ICL (in-context learning) cloning mode via `VoiceCloneForm`.
4. Submit to run real voice-clone inference (`ElBruno.QwenTTS.VoiceCloning.Pipeline.VoiceClonePipeline`) through `Services/VoiceClonePipelineService.cs`. Progress and errors surface via `VoiceCloneWorkflowStatus`; generated audio plays back with `VoiceSamplePlayer` and is saved to `wwwroot/generated`.
5. Cancel an in-flight generation at any time; the partially written output file is cleaned up.

This page requires the voice-cloning model download (same shared model directory as the home page) — it does not trigger downloads automatically.

## Notes

- Model availability and downloads are owned by server-side adapters, never by the browser or RCL. The sample uses the shared default model-directory conventions from `ModelDownloader` and `VoiceCloningDownloader`.
- The sample emits safe GenAI-oriented OpenTelemetry activities for model downloads, TTS synthesis, voice-clone embedding extraction, and voice-clone synthesis. They deliberately exclude prompts, reference audio, embeddings, and local paths.
- The `/voice-clone` page's speaker-embedding extraction and synthesis calls flow through the existing `QwenTtsTelemetry.StartVoiceCloning` instrumentation in `ElBruno.QwenTTS.VoiceCloning.Pipeline.VoiceClonePipeline` — no additional tracing was added. When the sample runs under `aspire start`, these show up as real GenAI traces in the Aspire dashboard.
- Replace demo handlers with production calls to your TTS and voice-cloning services.

## Aspire

`samples/BlazorQwenTtsDemo.AppHost` orchestrates this sample as its only executable resource. From the repository root, run:

```bash
aspire start --apphost samples/BlazorQwenTtsDemo.AppHost/BlazorQwenTtsDemo.AppHost.csproj --isolated --non-interactive
```

From `samples/BlazorQwenTtsDemo.AppHost`, run:

```bash
aspire start --isolated --non-interactive
```

Once running, open the sample's `/voice-clone` page and record/upload a reference clip, then generate cloned speech. The resulting speaker-embedding extraction and synthesis operations are visible as GenAI-tagged traces in the Aspire dashboard's Traces view (activity source `ElBruno.QwenTTS`, operations under `voice_clone.*`).
