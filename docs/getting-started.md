# Getting Started

## Quick Start (C# — Recommended)

The simplest way to get started is to let the library download the model files automatically:

```csharp
using ElBruno.QwenTTS.Pipeline;

// Models are downloaded automatically on first run (~5.5 GB)
var modelDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ElBruno.QwenTTS", "models");

using var pipeline = await TtsPipeline.CreateAsync(modelDir,
    progress: new Progress<string>(msg => Console.WriteLine(msg)));

await pipeline.SynthesizeAsync("Hello world!", "ryan", "hello.wav", "english");
Console.WriteLine("Done! Check hello.wav");
```

`TtsPipeline.CreateAsync` checks for missing files and downloads them from [HuggingFace](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX) automatically. Subsequent runs skip the download.

### Using the CLI app

```bash
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Hello world" --speaker ryan --language english --output hello.wav
```

The CLI also downloads models automatically if the `--model-dir` directory is missing files.

## Model Download API

You can also control the download process directly:

```csharp
using ElBruno.QwenTTS.Pipeline;

// Check if models are ready
if (!ModelDownloader.IsModelReady("./models"))
{
    // See what's missing
    var missing = ModelDownloader.GetMissingFiles("./models");
    Console.WriteLine($"{missing.Count} files need to be downloaded");

    // Download with progress
    await ModelDownloader.DownloadModelAsync("./models",
        progress: new Progress<ModelDownloadProgress>(p =>
            Console.WriteLine($"[{p.Current}/{p.Total}] {p.Message}")));
}
```

## Alternative: Python Setup Script

If you prefer to download models using Python (e.g., for CI environments or shared model storage):

```bash
python setup_environment.py
```

This script handles everything:

1. Checks prerequisites (Python >= 3.10, .NET 10 SDK)
2. Installs `huggingface_hub` (if missing)
3. Downloads ONNX models from HuggingFace (~5.5 GB)
4. Verifies all model files are present
5. Restores .NET NuGet packages

### Manual Python download

```bash
cd python
pip install huggingface_hub
python download_onnx_models.py --output-dir ../models
```

## Running Tests

```bash
dotnet test src/ElBruno.QwenTTS.Core.Tests
```

## Model Directory Structure

After downloading, the model directory looks like this:

```
models/
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
