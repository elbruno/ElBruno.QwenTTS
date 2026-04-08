# Issue #30: 1.7B Model Noise Diagnostic Test

This test project reproduces and diagnoses Issue #30, where the 1.7B model generates noise instead of clear speech.

## Purpose

Compares the working 0.6B model (baseline) with the problematic 1.7B model using:
- **English text**: "Hello, this is a test of the text to speech pipeline."
- **Chinese text** (from issue): "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"
- **Speaker**: vivian
- **Execution**: CPU-only (ONNX Runtime)

## Running the Test

From the repository root:

```bash
dotnet run --project tests/Issue30_17BNoiseTest/
```

**Note**: First run will download ~5.5 GB of 0.6B models and ~10 GB of 1.7B models from HuggingFace (elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX). Models are cached in `%LOCALAPPDATA%/ElBruno.QwenTTS/models/` (Windows) or `~/.local/share/ElBruno.QwenTTS/models/` (Linux/macOS).

## Output Files

The test generates 4 WAV files in the current directory:

1. `issue30_06b_english.wav` — 0.6B English (baseline)
2. `issue30_06b_chinese.wav` — 0.6B Chinese (baseline)
3. `issue30_17b_english.wav` — 1.7B English (test case)
4. `issue30_17b_chinese.wav` — 1.7B Chinese (reproduction of issue #30)

**Manual validation required**: Listen to the WAV files to compare audio quality. The 0.6B files should sound clear; the 1.7B files may sound like noise (as reported).

## Expected Behavior

If Issue #30 is reproduced:
- 0.6B files: Clear, intelligible speech
- 1.7B files: Noise or unintelligible audio

## Test Output

The test prints a comparison table showing:
- Model variant (0.6B vs 1.7B)
- Language (English vs Chinese)
- Output file name
- File size
- Synthesis duration

This helps identify if the issue is consistent across languages or specific to Chinese.
