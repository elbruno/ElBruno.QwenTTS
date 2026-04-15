"""
Download Qwen3-TTS model weights from Hugging Face Hub.

Models downloaded:
  - Qwen/Qwen3-TTS-0.6B-CustomVoice  (0.6B — CustomVoice variant)
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base    (0.6B — Base variant with voice cloning)
  - Qwen/Qwen3-TTS-1.7B-CustomVoice  (1.7B — CustomVoice variant with instruct control)
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base    (1.7B — Base variant with voice cloning)
  - Qwen/Qwen3-TTS-Tokenizer-12Hz    (vocoder / speech tokenizer, shared by all)

Usage:
  python download_models.py                              # Download all 0.6B models (default)
  python download_models.py --model customvoice          # Download 0.6B CustomVoice only
  python download_models.py --model customvoice-1.7b     # Download 1.7B CustomVoice only
  python download_models.py --model all-1.7b             # Download all 1.7B models
  python download_models.py --model everything           # Download all variants
"""

import argparse
import os
from pathlib import Path

from export_utils import configure_output_encoding
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

CUSTOM_VOICE_1_7B_MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "local_dir": "Qwen3-TTS-1.7B-CustomVoice",
    },
]

BASE_1_7B_MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_dir": "Qwen3-TTS-1.7B-Base",
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
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS model weights")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "all", "base", "customvoice",
            "all-1.7b", "base-1.7b", "customvoice-1.7b",
            "everything",
        ],
        default="all",
        help="Which model variant to download (default: all = 0.6B models + tokenizer)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models to {MODELS_DIR.resolve()}\n")

    models = list(SHARED_MODELS)
    if args.model in ("all", "customvoice", "everything"):
        models.extend(CUSTOM_VOICE_MODELS)
    if args.model in ("all", "base", "everything"):
        models.extend(BASE_MODELS)
    if args.model in ("all-1.7b", "customvoice-1.7b", "everything"):
        models.extend(CUSTOM_VOICE_1_7B_MODELS)
    if args.model in ("all-1.7b", "base-1.7b", "everything"):
        models.extend(BASE_1_7B_MODELS)

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
