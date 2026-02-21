# Qwen3-TTS ONNX Export Environment

Python environment for exporting Qwen3-TTS model components to ONNX format.

## Prerequisites

- Python 3.10 or later
- ~10 GB free disk space (for model weights)
- GPU recommended for export validation (CPU works but is slower)

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
python download_models.py
```

This downloads from Hugging Face Hub:
- **Qwen3-TTS-0.6B-CustomVoice** — the 0.6B-parameter TTS model with 9 built-in speakers
- **Qwen3-TTS-Tokenizer-12Hz** — the speech tokenizer / vocoder (converts audio codes ↔ waveforms)

Weights are saved to `models/` (gitignored). Total download is ~2–3 GB.

## Directory Layout

```
python/
├── requirements.txt        # Python dependencies
├── download_models.py      # Downloads model weights from HF Hub
├── models/                 # Downloaded model weights (gitignored)
│   ├── Qwen3-TTS-0.6B-CustomVoice/
│   └── Qwen3-TTS-Tokenizer-12Hz/
├── ARCHITECTURE.md         # Model architecture analysis
└── README.md               # This file
```

## Model Info

| Component | HF Repo | Size |
|-----------|---------|------|
| TTS LM | `Qwen/Qwen3-TTS-0.6B-CustomVoice` | ~1.8 GB |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | ~0.5 GB |

## Next Steps

After setup, see `ARCHITECTURE.md` for the model architecture analysis and ONNX export strategy.
