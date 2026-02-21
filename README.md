# Qwen3-TTS ONNX Pipeline + C# Console App

Run **Qwen3-TTS** text-to-speech locally from a C# .NET 10 console application using ONNX Runtime — no Python needed at inference time.

Pre-exported ONNX models are hosted on HuggingFace:
[**elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX**](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)

---

## Quick Start

```bash
python setup_environment.py
```

Then generate speech:

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hello, this is a test." --speaker ryan --language english --output hello.wav
```

## More Examples

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Welcome to the future of speech synthesis." --speaker serena --output welcome.wav

dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "This is a demo of local text to speech." --speaker vivian --language english --output demo.wav

dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Speaking with excitement and energy!" --speaker aiden --instruct "speak with excitement" --output excited.wav

dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "A calm and gentle narration." --speaker ryan --instruct "speak slowly and calmly" --output calm.wav
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Prerequisites](docs/prerequisites.md) | System requirements (Python, .NET, disk space) |
| [Getting Started](docs/getting-started.md) | Setup, model download, and first run |
| [CLI Reference](docs/cli-reference.md) | All command options, speakers, and examples |
| [Architecture](docs/architecture.md) | Pipeline design, model components, project structure |
| [Exporting Models](docs/exporting-models.md) | Re-exporting ONNX models from PyTorch weights |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Detailed Architecture](python/ARCHITECTURE.md) | Full tensor shapes, KV-cache, codebook structure |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Original model (PyTorch)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Pre-exported ONNX models](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)

## Squad Team

Morpheus (Lead), Trinity (ML Engineer), Neo (.NET Dev), Tank (Tester)
