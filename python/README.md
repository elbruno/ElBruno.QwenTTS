# Qwen3-TTS ONNX Export Environment

Python environment for exporting Qwen3-TTS model components to ONNX format.
Supports both the **0.6B** and **1.7B** model variants.

## Prerequisites

- Python 3.10 or later
- ~10 GB free disk space for 0.6B model weights (~25 GB for 1.7B)
- GPU recommended for export validation (CPU works but is slower)
- **1.7B export**: 16 GB+ VRAM recommended (or 32 GB+ system RAM for CPU export)

## Setup

### 1. Create a virtual environment

```bash
# From the python/ directory
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model weights

```bash
# Download 0.6B models (default — CustomVoice + Base + Tokenizer)
python download_models.py

# Download 1.7B CustomVoice model
python download_models.py --model customvoice-1.7b

# Download all 1.7B models (CustomVoice + Base + Tokenizer)
python download_models.py --model all-1.7b

# Download everything (0.6B + 1.7B + Tokenizer)
python download_models.py --model everything
```

This downloads from Hugging Face Hub:
- **Qwen3-TTS-0.6B-CustomVoice** — 0.6B-parameter TTS model with 9 built-in speakers
- **Qwen3-TTS-1.7B-CustomVoice** — 1.7B-parameter TTS with instruct control (emotion, rate, timbre)
- **Qwen3-TTS-Tokenizer-12Hz** — speech tokenizer / vocoder (shared by all variants)

## Directory Layout

```
python/
├── requirements.txt        # Python dependencies
├── download_models.py      # Downloads model weights from HF Hub
├── export_lm.py            # Export Talker LM + Code Predictor to ONNX
├── export_embeddings.py    # Extract embedding weights as .npy files
├── export_vocoder.py       # Export vocoder decoder to ONNX
├── extract_tokenizer.py    # Extract BPE tokenizer artifacts
├── reexport_lm_novmap.py   # Re-export with vmap-free masking (CustomVoice)
├── reexport_base_novmap.py # Re-export with vmap-free masking (Base)
├── models/                 # Downloaded model weights (gitignored)
│   ├── Qwen3-TTS-0.6B-CustomVoice/
│   ├── Qwen3-TTS-1.7B-CustomVoice/  (if downloaded)
│   └── Qwen3-TTS-Tokenizer-12Hz/
├── ARCHITECTURE.md         # Model architecture analysis
└── README.md               # This file
```

## Model Info

| Component | HF Repo | Size | Notes |
|-----------|---------|------|-------|
| TTS LM (0.6B) | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~1.8 GB | hidden=1024 |
| TTS LM (1.7B) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~5.4 GB | hidden=2048, instruct control |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | ~0.5 GB | Shared by all variants |

## Exporting ONNX Models

All export scripts read dimensions from the model's `config.json`, so the same
commands work for both 0.6B and 1.7B — just point `--model-dir` at the right
model directory.

### 0.6B Export (default)

```bash
python export_vocoder.py
python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/
python export_embeddings.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/embeddings
python extract_tokenizer.py
```

### 1.7B Export

```bash
python export_vocoder.py    # Vocoder is shared — same ONNX file
python export_lm.py --model-dir models/Qwen3-TTS-1.7B-CustomVoice --output-dir onnx_1.7b/
python export_embeddings.py --model-dir models/Qwen3-TTS-1.7B-CustomVoice --output-dir onnx_1.7b/embeddings
python extract_tokenizer.py  # Tokenizer is identical
```

### 1.7B Export Times & GPU Requirements

| Step | 0.6B (RTX 3090) | 1.7B (RTX 3090) | 1.7B (CPU, 64 GB) |
|------|-----------------|------------------|--------------------|
| Download | ~5 min | ~15 min | ~15 min |
| talker_prefill.onnx | ~2 min | ~5 min | ~30 min |
| talker_decode.onnx | ~2 min | ~5 min | ~30 min |
| code_predictor.onnx | ~30 sec | ~30 sec | ~2 min |
| vocoder.onnx | ~1 min | (shared) | (shared) |
| Embeddings + tokenizer | ~1 min | ~2 min | ~5 min |

> **Tip:** The 1.7B Talker LM models are ~3× larger than 0.6B. Expect ~5 GB ONNX
> files for each of talker_prefill and talker_decode. Total ONNX artifact size is
> ~15 GB for 1.7B (vs ~5.5 GB for 0.6B).

## Troubleshooting

### "RuntimeError: invalid unordered_map key" during ONNX export

This happens when transformers' vmap-based masking is incompatible with
`torch.onnx.export` tracing. The fix is automatic — `compat_patches.py`
(imported by all export scripts) registers vmap-free masking.

**If you still see this error:**
1. Ensure `compat_patches.py` is in the same directory as the export script
2. Check that you're running from the `python/` directory
3. Verify your transformers version: `pip show transformers`

### "ERROR: Model directory not found"

The export scripts require **locally downloaded** model weights. They do not
accept HuggingFace repo IDs directly. Download first:

```bash
python download_models.py                          # 0.6B models
python download_models.py --model customvoice-1.7b # 1.7B model
```

Then pass the local path:
```bash
python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/
```

### Compatibility Patches (`compat_patches.py`)

All export scripts import `compat_patches.py` which fixes incompatibilities
between the `qwen-tts` package and newer `transformers` versions (4.57+, 5.5+).
The patches handle: decorator API changes, RoPE init functions, SDPA mask
tracing, `torch.diff`/`cumsum` ONNX support, and vmap-free masking.

## Next Steps

After setup, see `ARCHITECTURE.md` for the model architecture analysis and ONNX export strategy.
