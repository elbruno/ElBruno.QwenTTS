# ElBruno.QwenTTS.Core — Library Reference

The **ElBruno.QwenTTS.Core** library provides the complete Qwen3-TTS inference pipeline for .NET applications. It handles tokenization, language model inference (via ONNX Runtime), vocoder decoding, and WAV file generation — all running locally with no external API calls.

## Installation

Add a project reference to `ElBruno.QwenTTS.Core`:

```xml
<ProjectReference Include="..\ElBruno.QwenTTS.Core\ElBruno.QwenTTS.Core.csproj" />
```

### Dependencies (included automatically)

| Package | Version | Purpose |
|---------|---------|---------|
| Microsoft.ML.OnnxRuntime | 1.24.2 | ONNX model inference |
| Microsoft.ML.Tokenizers | 2.0.0 | BPE text tokenization |
| NAudio | 2.2.1 | Audio processing |

## Quick Start

```csharp
using ElBruno.QwenTTS.Pipeline;

// Point to the directory containing the ONNX models
using var pipeline = new TtsPipeline("path/to/onnx_runtime");

// Generate speech
await pipeline.SynthesizeAsync(
    text: "Hello, welcome to the demo!",
    speaker: "ryan",
    outputPath: "output.wav"
);
```

## API Reference

### `TtsPipeline`

The main orchestrator class. Create one instance and reuse it — model loading is expensive.

```csharp
public sealed class TtsPipeline : IDisposable
```

#### Constructor

```csharp
public TtsPipeline(string modelDir)
```

| Parameter | Description |
|-----------|-------------|
| `modelDir` | Path to directory containing ONNX models, tokenizer, and embeddings |

The model directory must contain:

```
modelDir/
  tokenizer/          # BPE vocab and merges files
  embeddings/         # Speaker and codec embeddings (.npy files)
    config.json       # Model configuration
  talker_prefill.onnx
  talker_decode.onnx
  code_predictor.onnx
  vocoder.onnx
```

> **Tip:** Run `python setup_environment.py` from the repo root to download pre-exported models from [elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX). Models are placed in `python/onnx_runtime/`.

#### Properties

```csharp
public IReadOnlyCollection<string> Speakers { get; }
```

Returns available speaker voice names from the model. Current speakers: `ryan`, `serena`, `vivian`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.

#### Methods

```csharp
public async Task SynthesizeAsync(
    string text,
    string speaker,
    string outputPath,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `string` | The text to synthesize |
| `speaker` | `string` | Speaker voice name (e.g., `"ryan"`, `"serena"`) |
| `outputPath` | `string` | Path for the output WAV file (24 kHz, 16-bit PCM) |
| `language` | `string` | Language: `"english"`, `"spanish"`, `"chinese"`, `"japanese"`, `"korean"`, or `"auto"` (default) |
| `instruct` | `string?` | Optional voice style instruction (e.g., `"speak slowly and calmly"`) |
| `progress` | `IProgress<string>?` | Optional progress callback for real-time status updates |

```csharp
public void Dispose()
```

Releases all ONNX sessions, tokenizer, and embedding resources. Always dispose when done.

## Usage Examples

### Basic text-to-speech

```csharp
using ElBruno.QwenTTS.Pipeline;

using var pipeline = new TtsPipeline("python/onnx_runtime");
await pipeline.SynthesizeAsync("Hello world!", "ryan", "hello.wav");
```

### With language and voice style

```csharp
await pipeline.SynthesizeAsync(
    text: "Bienvenidos al futuro de la síntesis de voz.",
    speaker: "serena",
    outputPath: "bienvenidos.wav",
    language: "spanish",
    instruct: "speak with warmth and excitement"
);
```

### With progress reporting

```csharp
var progress = new Progress<string>(msg => Console.WriteLine($"  [{DateTime.Now:HH:mm:ss}] {msg}"));

await pipeline.SynthesizeAsync(
    text: "This is a test with progress tracking.",
    speaker: "vivian",
    outputPath: "progress_demo.wav",
    progress: progress
);
```

Output:

```
  [14:32:01] Tokenized input (47 tokens)
  [14:32:01] Running language model inference...
  [14:32:15] Generated 168 audio frames
  [14:32:15] Decoding waveform via vocoder...
  [14:32:16] Writing WAV file...
  [14:32:16] Saved progress_demo.wav (322560 samples, 13.44s)
```

### Batch processing multiple texts

```csharp
using var pipeline = new TtsPipeline("python/onnx_runtime");

var segments = new[]
{
    ("Welcome to the show!", "ryan"),
    ("Today we discuss AI agents.", "serena"),
    ("Let's dive right in.", "vivian"),
};

for (int i = 0; i < segments.Length; i++)
{
    var (text, speaker) = segments[i];
    await pipeline.SynthesizeAsync(text, speaker, $"segment_{i + 1:D2}.wav", "english");
    Console.WriteLine($"Generated segment {i + 1}");
}
```

### Listing available speakers

```csharp
using var pipeline = new TtsPipeline("python/onnx_runtime");

Console.WriteLine("Available voices:");
foreach (var speaker in pipeline.Speakers)
    Console.WriteLine($"  - {speaker}");
```

### ASP.NET / Blazor integration

For web apps, register the pipeline as a singleton service with thread-safe access:

```csharp
// Program.cs
builder.Services.AddSingleton<TtsPipelineService>();

// TtsPipelineService.cs
public class TtsPipelineService
{
    private readonly TtsPipeline _pipeline;
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    public TtsPipelineService(IConfiguration config)
    {
        var modelDir = config["TTS:ModelDir"] ?? "python/onnx_runtime";
        _pipeline = new TtsPipeline(modelDir);
    }

    public async Task<string> GenerateAsync(string text, string speaker,
        string language, string? instruct, IProgress<string>? progress)
    {
        await _semaphore.WaitAsync();
        try
        {
            var outputPath = Path.Combine("wwwroot/generated", $"{Guid.NewGuid()}.wav");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            await _pipeline.SynthesizeAsync(text, speaker, outputPath, language, instruct, progress);
            return $"/generated/{Path.GetFileName(outputPath)}";
        }
        finally { _semaphore.Release(); }
    }
}
```

## Architecture

The library has three layers:

```
Text → [TextTokenizer] → token IDs
     → [LanguageModel] → speech codec tokens (Talker LM + Code Predictor)
     → [Vocoder]       → 24 kHz WAV audio
```

| Component | Namespace | Description |
|-----------|-----------|-------------|
| `TtsPipeline` | `ElBruno.QwenTTS.Pipeline` | Orchestrator — the only class you need |
| `TextTokenizer` | `ElBruno.QwenTTS.Models` | BPE tokenization + prompt building |
| `LanguageModel` | `ElBruno.QwenTTS.Models` | Autoregressive LM with KV-cache (3 ONNX sessions) |
| `Vocoder` | `ElBruno.QwenTTS.Models` | Codec-to-waveform decoder |
| `EmbeddingStore` | `ElBruno.QwenTTS.Models` | Speaker/codec embedding lookups |
| `WavWriter` | `ElBruno.QwenTTS.Audio` | 24 kHz 16-bit PCM WAV writer |

For detailed model architecture (tensor shapes, KV-cache, codebook structure), see [Architecture](architecture.md).

## Requirements

- **.NET 10** (or later)
- **ONNX model files** — download via `python setup_environment.py` or from [HuggingFace](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)
- ~2 GB disk space for models
- ~4 GB RAM during inference
