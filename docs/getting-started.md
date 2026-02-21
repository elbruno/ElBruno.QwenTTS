# Getting Started

## One-Command Setup

```bash
python setup_environment.py
```

This single script handles everything:

1. Checks prerequisites (Python >= 3.10, .NET 10 SDK)
2. Installs `huggingface_hub` (if missing)
3. Downloads ONNX models from HuggingFace (~5.5 GB)
4. Verifies all model files are present
5. Restores .NET NuGet packages

## Manual Setup

If you prefer manual steps over the setup script:

### 1. Download ONNX models

```bash
cd python
pip install huggingface_hub
python download_onnx_models.py
```

Options:

```bash
python download_onnx_models.py --output-dir ./my-models
python download_onnx_models.py --repo-id your-user/your-repo
```

### 2. Build the C# app

```bash
dotnet restore src/QwenTTS/QwenTTS.csproj
dotnet build src/QwenTTS/QwenTTS.csproj
```

### 3. Generate speech

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hello world" --speaker ryan --language english --output hello.wav
```

## Model Directory Structure

After downloading, the model directory looks like this:

```
python/onnx_runtime/
├── talker_prefill.onnx + .data    (~1.7 GB — Talker LM prefill)
├── talker_decode.onnx + .data     (~1.7 GB — Talker LM decode)
├── code_predictor.onnx            (~420 MB — Code Predictor)
├── vocoder.onnx + .data           (~437 MB — Vocoder decoder)
├── embeddings/                    (text/codec embeddings, config, speaker IDs)
│   ├── config.json
│   ├── text_embedding.npy
│   ├── talker_codec_embedding.npy
│   ├── cp_codec_embedding_0..14.npy
│   ├── text_projection_fc*.npy
│   ├── codec_head_weight.npy
│   └── speaker_ids.json
└── tokenizer/
    ├── vocab.json
    └── merges.txt
```
