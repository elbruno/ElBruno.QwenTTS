"""Upload Base model ONNX files to HuggingFace."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

repo_id = "elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX"
print(f"Creating repo {repo_id}...")
create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

api = HfApi()
onnx_dir = Path("onnx_base")

# Upload ONNX models
onnx_files = sorted(list(onnx_dir.glob("*.onnx")) + list(onnx_dir.glob("*.onnx.data")))
print(f"Uploading {len(onnx_files)} ONNX files...")
for f in onnx_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  -> {f.name} ({size_mb:.1f} MB)")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id)

# Upload embeddings
emb_dir = onnx_dir / "embeddings"
emb_files = sorted(emb_dir.iterdir())
print(f"Uploading {len(emb_files)} embedding files...")
for f in emb_files:
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  -> embeddings/{f.name} ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"embeddings/{f.name}",
            repo_id=repo_id,
        )

# Upload tokenizer
tok_dir = onnx_dir / "tokenizer"
print("Uploading tokenizer files...")
for name in ["vocab.json", "merges.txt"]:
    f = tok_dir / name
    if f.exists():
        print(f"  -> tokenizer/{name}")
        api.upload_file(
            path_or_fileobj=str(f), path_in_repo=f"tokenizer/{name}", repo_id=repo_id
        )

# Upload README
readme = """---
license: apache-2.0
tags:
  - onnx
  - tts
  - qwen3-tts
  - text-to-speech
  - voice-cloning
  - ecapa-tdnn
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
---

# Qwen3-TTS 12Hz 0.6B Base — ONNX (Voice Cloning)

ONNX export of [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) for local inference with C# / ONNX Runtime. Includes ECAPA-TDNN speaker encoder for 3-second voice cloning.

## Files

| File | Description | Size |
|------|-------------|------|
| `speaker_encoder.onnx` + `.data` | ECAPA-TDNN speaker encoder | ~34 MB |
| `talker_prefill.onnx` + `.data` | Talker LM prefill (28 layers) | ~1.7 GB |
| `talker_decode.onnx` + `.data` | Talker LM single-step decode | ~1.7 GB |
| `code_predictor.onnx` | Code Predictor (5 layers, 15 groups) | ~440 MB |
| `vocoder.onnx` | Vocoder decoder (24kHz output) | ~2.7 MB |
| `embeddings/` | Text/codec embeddings as .npy + config | ~1.4 GB |
| `tokenizer/` | BPE tokenizer (vocab.json, merges.txt) | ~4 MB |

## Usage with C#

```bash
dotnet add package ElBruno.QwenTTS.VoiceCloning
```

```csharp
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

var cloner = await VoiceClonePipeline.CreateAsync();
await cloner.SynthesizeAsync("Hello world!", "reference.wav", "output.wav", "english");
```

## Architecture

- **Speaker Encoder**: ECAPA-TDNN, 128 mel bins input, 1024-dim output
- **Talker**: 28 transformer layers, 16 attn heads, 8 KV heads, hidden=1024
- **Code Predictor**: 5 layers, generates codebook groups 1-15
- **Vocoder**: RVQ dequantize → transformer → BigVGAN decoder, 12Hz → 24kHz

## License

Apache-2.0 (same as base model)
"""
print("  -> README.md")
api.upload_file(
    path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id
)

print(f"\nAll files uploaded to https://huggingface.co/{repo_id}")
