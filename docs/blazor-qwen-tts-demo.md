# BlazorQwenTtsDemo Sample App

The sample app at `samples/BlazorQwenTtsDemo` demonstrates all components from `ElBruno.QwenTTS.BlazorComponents`.

## Run

```bash
dotnet run --project samples/BlazorQwenTtsDemo
```

Open the local URL printed by ASP.NET Core and use the home page to test:

1. `ModelDownloadStatus` for model state UI.
2. `TtsInputPanel` for text input and synthesis trigger.
3. `SynthesisProgressBar` for progress feedback.
4. `VoiceCloningSamplePicker` for reference sample selection/upload.
5. `VoiceSamplePlayer` for generated/reference audio playback.

## Notes

- The sample is UI-focused and demonstrates integration patterns.
- Replace demo handlers with production calls to your TTS and voice-cloning services.
