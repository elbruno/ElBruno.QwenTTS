# Issue #30: 1.7B Vocoder Re-Export Guide

## Problem Summary

**Issue #30:** The 1.7B model generates noise and truncated audio output (fixed at 0.80 seconds / 19,200 samples) regardless of input.

**Root Cause:** The 1.7B `vocoder.onnx` model was exported with a **fixed timestep dimension** (T=10 baked into the model graph). This causes the vocoder to always decode exactly 10 timesteps, producing 10 × 1,920 = 19,200 audio samples.

**Evidence:**
- Regardless of input size (226 or 795 frames from the language model), the vocoder always outputs 19,200 samples
- 19,200 / 1,920 (samples per timestep) = 10 timesteps (hardcoded)
- The 0.6B vocoder works correctly because it was exported with **dynamic axes** and produces T × 1,920 samples as expected

## C# Fixes Applied

These fixes are on branch `squad/30-fix-17b-model` and address the issues in the .NET library:

### 1. ModelDownloader: Variant-Specific File Lists
- `vocoder.onnx.data` is **0.6B-only** (not needed for 1.7B if vocoder ≤512 MB)
- `code_predictor.onnx.data` is **1.7B-only** (large model file split)
- Prevents failed downloads and disk space waste

### 2. EmbeddingStore.GetSpeakerEmbedding
- Changed from hardcoded `1024` to `_hiddenSize` (set per model variant)
- Ensures correct embedding dimensions for both 0.6B (1024) and 1.7B (1024, but used differently)

### 3. LanguageModel.BuildPrefillEmbedding
- For 1.7B: Zero-pad the speaker embedding from 1024 → 2048 dims to match model input
- 0.6B: No padding needed (1024-dim input as-is)

### 4. LanguageModel: Dynamic Attention Mask Buffer Sizing
- Attention mask buffer no longer assumes fixed sequence length
- Scales with actual input size to prevent OOM or dimension mismatches

### 5. Vocoder.Decode Output Size Validation
- Added validation that output size matches expected value: `actual_samples == T × 1,920`
- Clear error message when vocoder produces unexpected output (helps catch export issues early)

## Vocoder Re-Export Steps

The 1.7B vocoder must be re-exported with **dynamic timestep axes**. Perform these steps on a machine with **PyTorch**, **transformers**, and the **Qwen3-TTS-Tokenizer-12Hz** model available.

### Step 1: Run the Export Script
```bash
cd python/
python export_vocoder.py --timesteps 20 --output-dir ./onnx_models_17b/
```

This exports the vocoder with:
- **Dynamic axes** for `num_timesteps` (not fixed to 10)
- Test timestep value of 20 (ensures the model accepts variable input sizes)
- Output directory: `./onnx_models_17b/vocoder.onnx`

### Step 2: Validate Dynamic Axes
```bash
python validate_vocoder.py --onnx-path ./onnx_models_17b/vocoder.onnx
```

Verify that:
- Model input shape has dynamic axis named `num_timesteps` (not a fixed value like `10`)
- Model accepts multiple timestep sizes without error
- Output shape is correctly computed as `[batch_size, num_timesteps * 1920]`

### Step 3: Upload to HuggingFace
Replace the old vocoder in the 1.7B repo:

```bash
huggingface-cli upload elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX \
  ./onnx_models_17b/vocoder.onnx vocoder.onnx
```

Ensure you are authenticated to HuggingFace:
```bash
huggingface-cli login
# Enter your token when prompted
```

## HuggingFace Repository File Differences

| File | 0.6B Repo | 1.7B Repo | Notes |
|------|-----------|-----------|-------|
| `vocoder.onnx` | ✅ | ✅ (needs re-export) | Main vocoder; 1.7B version was fixed at T=10 |
| `vocoder.onnx.data` | ✅ | ❌ | 0.6B only; not needed if vocoder fits in single ONNX file |
| `code_predictor.onnx.data` | ❌ | ✅ | 1.7B only; large language model split into separate file |
| `talker_prefill.onnx` | ✅ | ✅ | Language model prefill phase |
| `talker_prefill.onnx.data` | ✅ | ✅ | Language model prefill weights (split) |
| `talker_decode.onnx` | ✅ | ✅ | Language model decode phase |
| `talker_decode.onnx.data` | ✅ | ✅ | Language model decode weights (split) |

## Verification After Re-Export

After uploading the new vocoder to HuggingFace, verify the fix with the test application:

```bash
dotnet run --project tests/Issue30_17BNoiseTest/
```

1. When prompted for 0.6B test, answer **n** (skip; 0.6B already works)
2. When prompted for 1.7B test, answer **y** (verify the fix)

**Expected Output for 1.7B:**

- **English synthesis:**
  - ~50–70 output frames (proportional to input text, not fixed to 10)
  - ~100–140 KB WAV file (proportional to frames)
  - Clear, intelligible speech (no noise or distortion)

- **Chinese synthesis:**
  - ~60–80 output frames
  - ~120–160 KB WAV file
  - Clear, intelligible speech

**If output is still 19,200 samples / 0.80s:**
- Vocoder re-export did not use dynamic axes
- Re-run `export_vocoder.py` and verify `validate_vocoder.py` output
- Check that the uploaded `vocoder.onnx` has the correct axes (re-validate after upload if needed)
