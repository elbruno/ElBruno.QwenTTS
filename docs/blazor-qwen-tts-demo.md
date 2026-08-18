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

## Notes

- Model availability and downloads are owned by server-side adapters, never by the browser or RCL. The sample uses the shared default model-directory conventions from `ModelDownloader` and `VoiceCloningDownloader`.
- The sample emits safe GenAI-oriented OpenTelemetry activities for model downloads, TTS synthesis, voice-clone embedding extraction, and voice-clone synthesis. They deliberately exclude prompts, reference audio, embeddings, and local paths.
- Replace demo handlers with production calls to your TTS and voice-cloning services.

## Aspire

`ElBruno.QwenTTS.AppHost` orchestrates this sample as its only executable resource:

```bash
aspire start --isolated --non-interactive
```
