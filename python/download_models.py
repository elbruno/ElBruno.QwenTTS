"""
Download Qwen3-TTS model weights from Hugging Face Hub.

Models downloaded:
  - Qwen/Qwen3-TTS-0.6B-CustomVoice  (the language model)
  - Qwen/Qwen3-TTS-Tokenizer-12Hz    (the vocoder / speech tokenizer)

Usage:
  python download_models.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-0.6B-CustomVoice",
        "local_dir": "Qwen3-TTS-0.6B-CustomVoice",
    },
    {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "local_dir": "Qwen3-TTS-Tokenizer-12Hz",
    },
]

MODELS_DIR = Path(__file__).parent / "models"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models to {MODELS_DIR.resolve()}\n")

    for model in MODELS:
        dest = MODELS_DIR / model["local_dir"]
        print(f"--- {model['repo_id']} → {dest}")
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(dest),
        )
        print(f"    ✓ Done\n")

    print("All models downloaded.")


if __name__ == "__main__":
    main()
