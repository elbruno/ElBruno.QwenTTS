# ElBruno.QwenTTS — Library Reference

[![NuGet](https://img.shields.io/nuget/v/ElBruno.QwenTTS.svg)](https://www.nuget.org/packages/ElBruno.QwenTTS)

The **ElBruno.QwenTTS** library provides the complete Qwen3-TTS inference pipeline for .NET applications. It handles tokenization, language model inference (via ONNX Runtime), vocoder decoding, and WAV file generation — all running locally with no external API calls.

## Installation

### NuGet Package (recommended)

```bash
dotnet add package ElBruno.QwenTTS
```

### Project Reference (for source-level development)

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

// CreateAsync downloads models automatically if missing (~5.5 GB on first run)
using var pipeline = await TtsPipeline.CreateAsync("models");

// Generate speech
await pipeline.SynthesizeAsync(
    text: "Hello, welcome to the demo!",
    speaker: "ryan",
    outputPath: "output.wav"
);

// Or keep the audio in memory
var pcm = await pipeline.SynthesizeToPcmAsync(
    text: "Return PCM samples without an output file.",
    speaker: "ryan");
var wav = await pipeline.SynthesizeWavAsync(
    text: "Return WAV bytes without an output file.",
    speaker: "ryan");
```

> **Note:** Use `TtsPipeline.CreateAsync()` for automatic model management. Use `new TtsPipeline(modelDir)` only when you know the models are already present.

> **Performance note:** Create one pipeline or client and reuse it. ONNX sessions now stay loaded for the lifetime of the pipeline, which avoids reloading model graphs on every request.

## API Reference

### `TtsPipeline`

The main orchestrator class. Create one instance and reuse it — model loading is expensive.

```csharp
public sealed class TtsPipeline : IDisposable
```

#### Static Factory (recommended)

```csharp
public static async Task<TtsPipeline> CreateAsync(
    string modelDir,
    string repoId = ModelDownloader.DefaultRepoId,
    IProgress<string>? progress = null,
    int maxConcurrency = 1,
    CancellationToken cancellationToken = default)
```

Creates a `TtsPipeline`, automatically downloading missing model files from HuggingFace. This is the recommended way to create a pipeline.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modelDir` | `string` | Directory to store/load model files |
| `repoId` | `string` | HuggingFace repo ID (default: `elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX`) |
| `progress` | `IProgress<string>?` | Optional progress callback for download status |
| `maxConcurrency` | `int` | Maximum number of concurrent syntheses allowed per shared pipeline instance |
| `cancellationToken` | `CancellationToken` | Cancellation token |

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

> **Tip:** Use `TtsPipeline.CreateAsync()` instead — it downloads these files automatically. The manual constructor is for advanced scenarios where you manage model files yourself.

#### Properties

```csharp
public IReadOnlyCollection<string> Speakers { get; }
```

Returns available speaker voice names from the model. Current speakers: `ryan`, `serena`, `vivian`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.

#### Methods

```csharp
public async Task<TtsAudioResult> SynthesizeToPcmAsync(
    string text,
    string speaker,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null,
    CancellationToken cancellationToken = default)
```

Returns normalized float PCM samples in the `[-1.0, 1.0]` range, along with sample metadata, derived duration, and request metrics.

| Property | Type | Description |
|----------|------|-------------|
| `Samples` | `ReadOnlyMemory<float>` | PCM samples in the `[-1.0, 1.0]` range |
| `SampleRate` | `int` | Output sample rate (24 kHz by default) |
| `Channels` | `int` | Number of output channels (currently `1`) |
| `BitsPerSample` | `int` | WAV PCM bit depth used when encoding bytes (`16`) |
| `SampleCount` | `int` | Total PCM sample count |
| `Duration` | `TimeSpan` | Duration derived from sample count and format metadata |
| `Metrics` | `TtsSynthesisMetrics` | Queue, first-audio, and total synthesis timings |

```csharp
public async Task<ReadOnlyMemory<byte>> SynthesizeWavAsync(
    string text,
    string speaker,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null,
    CancellationToken cancellationToken = default)
```

Returns a complete in-memory WAV payload without requiring an output file path.

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
| `language` | `string` | Language: `"english"`, `"spanish"`, `"chinese"`, `"japanese"`, `"korean"`, `"russian"`, or `"auto"` (default) |
| `instruct` | `string?` | Optional voice style instruction (e.g., `"speak slowly and calmly"`) |
| `progress` | `IProgress<string>?` | Optional progress callback for real-time status updates |

```csharp
public async Task<TtsSynthesisMetrics> SynthesizeWithMetricsAsync(
    string text,
    string speaker,
    string outputPath,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null,
    CancellationToken cancellationToken = default)
```

Returns queue latency, first-audio latency, total latency, generated frame count, and output sample count for the request.

```csharp
public async IAsyncEnumerable<TextToSpeechStreamingUpdate> SynthesizeStreamingAsync(
    string text,
    string speaker,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null,
    int maxChunkBytes = 32 * 1024,
    CancellationToken cancellationToken = default)
```

Streams ordered WAV chunks. The first update is `SessionOpen` and carries format metadata (`MediaType`, `SampleRate`, `Channels`, `BitsPerSample`) plus `IsProgressive`, which is currently `false` because the pipeline emits ordered post-generation chunks after vocoder decode completes.

```csharp
public IAsyncEnumerable<TextToSpeechStreamingUpdate> GetStreamingAudioAsync(
    string text,
    string speaker,
    string language = "auto",
    string? instruct = null,
    IProgress<string>? progress = null,
    int maxChunkBytes = 32 * 1024,
    CancellationToken cancellationToken = default)
```

Convenience alias for callers that prefer a get-style name.

```csharp
public void Dispose()
```

Releases all ONNX sessions, tokenizer, and embedding resources. Always dispose when done.

### `QwenTextToSpeechClient`

`QwenTextToSpeechClient` keeps the existing high-level convenience APIs and also implements `Microsoft.Extensions.AI.ITextToSpeechClient` through an adapter layer over `TtsPipeline`.

```csharp
public sealed class QwenTextToSpeechClient :
    ITextToSpeechClient,
    Microsoft.Extensions.AI.ITextToSpeechClient
```

The MEAI surface maps directly onto the in-memory pipeline methods, so `GetAudioAsync()` returns WAV data without a temporary file and `GetStreamingAudioAsync()` forwards ordered streaming updates through the MEAI response-update contract.

```csharp
using ElBruno.QwenTTS.Pipeline;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

var services = new ServiceCollection();
services.AddQwenTextToSpeechClient(options =>
{
    options.ModelVariant = QwenModelVariant.Qwen17B;
    options.ExecutionProvider = ExecutionProvider.DirectML;
    options.InstructText = "Read with warmth";
});

using var provider = services.BuildServiceProvider();
var client = provider.GetRequiredService<ITextToSpeechClient>();

var response = await client.GetAudioAsync(
    "Hello from MEAI",
    new TextToSpeechOptions
    {
        VoiceId = "serena",
        Language = "english",
        AdditionalProperties = new()
        {
            [QwenTextToSpeechMetadataKeys.Instruct] = "speak with excitement"
        }
    });

var wav = (DataContent)response.Contents.Single();
await File.WriteAllBytesAsync("meai.wav", wav.Data.ToArray());
```

#### MEAI Metadata

The adapter populates `AdditionalProperties` on MEAI responses and streaming updates with the following keys:

| Key | Meaning |
|-----|---------|
| `elbruno.qwentts.variant` | Model variant (`0.6B`, `1.7B`) |
| `elbruno.qwentts.speaker` | Resolved speaker/voice |
| `elbruno.qwentts.language` | Resolved language |
| `elbruno.qwentts.instruct` | Instruct text when supplied |
| `elbruno.qwentts.voice_cloning` | Always `false` for this adapter |
| `elbruno.qwentts.execution_provider` | Configured execution provider (`Cpu`, `Cuda`, `DirectML`) |

#### MEAI Limitations

- Output format is WAV only (`audio/wav`).
- `Speed`, `Pitch`, and `Volume` are rejected with a clear capability error.
- Voice-cloning input is rejected with a clear capability error; use `ElBruno.QwenTTS.VoiceCloning` for reference-audio synthesis.

### `ModelDownloader`

Static utility class for downloading and verifying ONNX model files.

```csharp
public sealed class ModelDownloader
```

#### Properties

```csharp
// Default shared model directory (%LOCALAPPDATA%/ElBruno.QwenTTS/models)
// All apps share this location to avoid duplicate downloads
public static string DefaultModelDir { get; }

public const string DefaultRepoId = "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX";
```

#### Methods

```csharp
// Check if all model files are present (uses DefaultModelDir if null)
public static bool IsModelDownloaded(string? modelDir = null);

// Get list of missing files
public static IReadOnlyList<string> GetMissingFiles(string? modelDir = null);

// Download with byte-level progress
public static async Task DownloadModelAsync(
    string? modelDir = null,
    string repoId = DefaultRepoId,
    IProgress<ModelDownloadProgress>? progress = null,
    CancellationToken cancellationToken = default);

// Ensure models are present (download if needed), return modelDir
public static async Task<string> EnsureModelAsync(
    string? modelDir = null,
    string repoId = DefaultRepoId,
    IProgress<ModelDownloadProgress>? progress = null,
    CancellationToken cancellationToken = default);
```

### `ModelDownloadProgress`

```csharp
public record ModelDownloadProgress(
    int CurrentFile, int TotalFiles, string? FileName, string Message,
    long BytesDownloaded, long TotalBytes)
{
    public double FilePercentage { get; }  // File-level 0-100
    public double BytePercentage { get; }  // Byte-level 0-100 for current file
}
```

### `TextToSpeechResponse`

The high-level `QwenTextToSpeechClient` returns a `TextToSpeechResponse` with audio bytes plus latency metadata:

```csharp
public sealed class TextToSpeechResponse
{
    public required byte[] AudioData { get; init; }
    public string MediaType { get; init; }
    public int SampleRate { get; init; }
    public string ModelId { get; init; }
    public TtsSynthesisMetrics Metrics { get; init; }
}
```

### `TextToSpeechStreamingUpdate`

The streaming client and pipeline emit lifecycle updates with format metadata and ordered audio chunks:

```csharp
public sealed class TextToSpeechStreamingUpdate
{
    public required TextToSpeechUpdateKind Kind { get; init; }
    public byte[]? AudioData { get; init; }
    public int? SampleRate { get; init; }
    public string? MediaType { get; init; }
    public int? Channels { get; init; }
    public int? BitsPerSample { get; init; }
    public bool IsProgressive { get; init; }
    public TtsSynthesisMetrics? Metrics { get; init; }
}
```

Concatenating every `AudioChunk.AudioData` in order reconstructs a valid WAV file.

### `QwenTtsOptions`

DI registration accepts model, execution-provider, instruct, and concurrency settings:

```csharp
builder.Services.AddQwenTextToSpeechClient(options =>
{
    options.ModelVariant = QwenModelVariant.Qwen17B;
    options.ExecutionProvider = ExecutionProvider.Cuda;
    options.InstructText = "Read with warmth";
    options.MaxConcurrency = 2;
});
```

### Shared Model Directory

All apps using `ElBruno.QwenTTS.Core` share the same default model directory:

- **Windows:** `%LOCALAPPDATA%\ElBruno.QwenTTS\models`
- **Linux/macOS:** `~/.local/share/ElBruno.QwenTTS/models`

This means models are downloaded once and reused by CLI, Web, and any custom app. Override with a custom path if needed.

## Usage Examples

### Basic text-to-speech

```csharp
using ElBruno.QwenTTS.Pipeline;

// Auto-downloads models on first run
using var pipeline = await TtsPipeline.CreateAsync("models");
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

### Russian speech

```csharp
await pipeline.SynthesizeAsync(
    text: "Привет, это тест синтеза речи.",
    speaker: "ryan",
    outputPath: "russian.wav",
    language: "russian"
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

### Streaming to an HTTP response or file sink

```csharp
await foreach (var update in pipeline.GetStreamingAudioAsync(
    text: "Stream this clip in ordered WAV chunks.",
    speaker: "ryan",
    language: "english",
    cancellationToken: cancellationToken))
{
    if (update.Kind == TextToSpeechUpdateKind.SessionOpen)
        Console.WriteLine($"{update.MediaType} {update.SampleRate}Hz progressive={update.IsProgressive}");

    if (update.Kind == TextToSpeechUpdateKind.AudioChunk)
        await destination.WriteAsync(update.AudioData!, cancellationToken);
}
```

### Batch processing multiple texts

```csharp
using var pipeline = await TtsPipeline.CreateAsync("models");

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
using var pipeline = await TtsPipeline.CreateAsync("models");

Console.WriteLine("Available voices:");
foreach (var speaker in pipeline.Speakers)
    Console.WriteLine($"  - {speaker}");
```

### ASP.NET / Blazor integration

For web apps, register the pipeline as a singleton service and let the shared pipeline manage bounded concurrency:

```csharp
// Program.cs
builder.Services.AddSingleton<TtsPipelineService>();

// TtsPipelineService.cs
public class TtsPipelineService
{
    private readonly TtsPipeline _pipeline;

    public TtsPipelineService(IConfiguration config)
    {
        var modelDir = config["TTS:ModelDir"] ?? "python/onnx_runtime";
        _pipeline = new TtsPipeline(modelDir, maxConcurrency: 2);
    }

    public async Task<string> GenerateAsync(string text, string speaker,
        string language, string? instruct, IProgress<string>? progress)
    {
        var outputPath = Path.Combine("wwwroot/generated", $"{Guid.NewGuid()}.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        await _pipeline.SynthesizeAsync(text, speaker, outputPath, language, instruct, progress);
        return $"/generated/{Path.GetFileName(outputPath)}";
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
