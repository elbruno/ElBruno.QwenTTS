"""
Download ONNX model artifacts from HuggingFace Hub.

Downloads all files needed to run the C# QwenTTS app:
  - ONNX models (talker_prefill, talker_decode, code_predictor, vocoder)
  - Embedding .npy files + config
  - Tokenizer artifacts (vocab.json, merges.txt)

No HF token needed — the repo is public.

Usage:
  python download_onnx_models.py
  python download_onnx_models.py --repo-id elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX --output-dir ./models
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_REPO_ID = "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX"


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models from HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_runtime",
        help="Local directory to save models (default: onnx_runtime)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models from {args.repo_id} → {output_dir.resolve()}")

    # List all files in the repo
    files = list_repo_files(args.repo_id)
    # Filter out README
    files = [f for f in files if f != "README.md" and not f.startswith(".")]

    total = len(files)
    for idx, filename in enumerate(sorted(files), 1):
        # Determine local path
        local_path = output_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"  [{idx}/{total}] ✓ {filename} (already exists)")
            continue

        print(f"  [{idx}/{total}] ↓ {filename} ...")
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=str(output_dir),
        )

    print(f"\n✅ All models downloaded to {output_dir.resolve()}")
    print(f"\nRun the C# app:")
    print(f"  dotnet run --project src/QwenTTS -- --model-dir {output_dir} --text \"Hello world\" --speaker ryan --language english")


if __name__ == "__main__":
    main()
