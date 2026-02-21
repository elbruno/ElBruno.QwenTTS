"""
Upload ONNX model artifacts to HuggingFace Hub.

Creates a new HF repo (or uses existing) and uploads:
  - ONNX models (talker_prefill, talker_decode, code_predictor, vocoder)
  - Embedding .npy files + config
  - Tokenizer artifacts (vocab.json, merges.txt)

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   # needs write token from https://huggingface.co/settings/tokens

Usage:
  python upload_to_hf.py --repo-id elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX
  python upload_to_hf.py --repo-id elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX --private
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="onnx_models",
        help="Directory containing ONNX models and embeddings (default: onnx_models)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="tokenizer_artifacts",
        help="Directory containing tokenizer files (default: tokenizer_artifacts)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (default: public)",
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    api = HfApi()

    # Create repo (idempotent)
    print(f"Creating repo {args.repo_id} ...")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # Upload ONNX models (root level)
    onnx_files = list(onnx_dir.glob("*.onnx")) + list(onnx_dir.glob("*.onnx.data"))
    print(f"\nUploading {len(onnx_files)} ONNX files ...")
    for f in sorted(onnx_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  ↑ {f.name} ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo_id,
        )

    # Upload embeddings
    embeddings_dir = onnx_dir / "embeddings"
    if embeddings_dir.exists():
        emb_files = list(embeddings_dir.iterdir())
        print(f"\nUploading {len(emb_files)} embedding files ...")
        for f in sorted(emb_files):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  ↑ embeddings/{f.name} ({size_mb:.1f} MB)")
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"embeddings/{f.name}",
                    repo_id=args.repo_id,
                )

    # Upload tokenizer (only vocab.json and merges.txt needed for C# runtime)
    print(f"\nUploading tokenizer files ...")
    for name in ["vocab.json", "merges.txt"]:
        f = tokenizer_dir / name
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ↑ tokenizer/{name} ({size_mb:.1f} MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"tokenizer/{name}",
                repo_id=args.repo_id,
            )

    # Upload a README model card
    readme = f"""---
license: apache-2.0
tags:
  - onnx
  - tts
  - qwen3-tts
  - text-to-speech
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
---

# Qwen3-TTS 12Hz 0.6B CustomVoice — ONNX

ONNX export of [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) for local inference with C# / ONNX Runtime.

## Files

| File | Description | Size |
|------|-------------|------|
| `talker_prefill.onnx` + `.data` | Talker LM prefill (28 layers) | ~1.7 GB |
| `talker_decode.onnx` + `.data` | Talker LM single-step decode | ~1.7 GB |
| `code_predictor.onnx` | Code Predictor (5 layers, 15 groups) | ~420 MB |
| `vocoder.onnx` + `.data` | Vocoder decoder (24kHz output) | ~437 MB |
| `embeddings/` | Text/codec embeddings as .npy + config | ~1.4 GB |
| `tokenizer/` | BPE tokenizer (vocab.json, merges.txt) | ~4 MB |

## Usage with C#

```bash
# Clone the app repo
git clone https://github.com/elbruno/ElBruno.QwenTTS.git
cd ElBruno.QwenTTS

# Download models
python python/download_onnx_models.py --repo-id {args.repo_id}

# Run
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hello world" --speaker ryan --language english
```

## Architecture

- **Talker**: 28 transformer layers, 16 attn heads, 8 KV heads, hidden=1024
- **Code Predictor**: 5 layers, generates codebook groups 1-15
- **Vocoder**: RVQ dequantize → transformer → BigVGAN decoder, 12Hz → 24kHz (1920× upsample)
- **KV Cache**: Decode uses stacked format (num_layers, B, num_kv_heads, T, head_dim)
- **Speakers**: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan

## License

Apache-2.0 (same as base model)
"""
    print("  ↑ README.md")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )

    print(f"\n✅ All files uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
