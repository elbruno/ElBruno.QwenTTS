"""
Download Qwen3-TTS model weights from Hugging Face Hub.

Models downloaded:
  - Qwen/Qwen3-TTS-0.6B-CustomVoice  (the language model — CustomVoice variant)
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base    (the language model — Base variant with voice cloning)
  - Qwen/Qwen3-TTS-Tokenizer-12Hz    (the vocoder / speech tokenizer, shared)

Usage:
  python download_models.py                    # Download all models
  python download_models.py --model base       # Download Base model only
  python download_models.py --model customvoice # Download CustomVoice model only
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


CUSTOM_VOICE_MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "local_dir": "Qwen3-TTS-0.6B-CustomVoice",
    },
]

BASE_MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "local_dir": "Qwen3-TTS-0.6B-Base",
    },
]

SHARED_MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "local_dir": "Qwen3-TTS-Tokenizer-12Hz",
    },
]

MODELS_DIR = Path(__file__).parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS model weights")
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "base", "customvoice"],
        default="all",
        help="Which model variant to download (default: all)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models to {MODELS_DIR.resolve()}\n")

    models = list(SHARED_MODELS)
    if args.model in ("all", "customvoice"):
        models.extend(CUSTOM_VOICE_MODELS)
    if args.model in ("all", "base"):
        models.extend(BASE_MODELS)

    for model in models:
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
