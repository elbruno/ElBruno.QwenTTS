using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using ElBruno.QwenTTS.Audio;
using ElBruno.QwenTTS.Models;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Orchestrates the full TTS pipeline: text → tokenize → LM → vocoder → WAV.
/// </summary>
public sealed class TtsPipeline : ITtsPipeline
{
    private readonly TextTokenizer _tokenizer;
    private readonly LanguageModel _languageModel;
    private readonly Vocoder _vocoder;
    private readonly EmbeddingStore _embeddings;
    private readonly QwenModelVariant _variant;
    private readonly TtsConcurrencyGate _concurrencyGate;
    private const string StreamingMediaType = "audio/wav";

    private sealed record SynthesisArtifacts(float[] Waveform, TtsSynthesisMetrics Metrics);

    /// <summary>
    /// Creates a TtsPipeline from a local model directory.
    /// </summary>
    /// <param name="modelDir">Directory containing ONNX models, embeddings, and tokenizer.</param>
    /// <param name="sessionOptionsFactory">Optional factory for ONNX Runtime session options (e.g., for GPU acceleration). When null, uses CPU with max optimization.</param>
    /// <param name="vocoderSessionOptionsFactory">Optional separate factory for the vocoder model. Useful when GPU EP doesn't support vocoder ops (e.g., DirectML). When null, uses sessionOptionsFactory.</param>
    /// <param name="variant">Model size variant. Used to determine feature support (e.g., instruction control).</param>
    /// <param name="maxConcurrency">Maximum number of synthesis requests allowed to run concurrently per pipeline instance.</param>
    public TtsPipeline(string modelDir, Func<SessionOptions>? sessionOptionsFactory = null, Func<SessionOptions>? vocoderSessionOptionsFactory = null, QwenModelVariant variant = QwenModelVariant.Qwen06B, int maxConcurrency = 1)
    {
        var tokenizerDir = Path.Combine(modelDir, "tokenizer");
        var embeddingsDir = Path.Combine(modelDir, "embeddings");
        var configPath = Path.Combine(embeddingsDir, "config.json");

        _variant = variant;
        _concurrencyGate = new TtsConcurrencyGate(maxConcurrency);
        _tokenizer = new TextTokenizer(tokenizerDir);
        _embeddings = new EmbeddingStore(embeddingsDir, configPath);
        _languageModel = new LanguageModel(modelDir, _embeddings, sessionOptionsFactory);
        _vocoder = new Vocoder(Path.Combine(modelDir, "vocoder.onnx"), vocoderSessionOptionsFactory ?? sessionOptionsFactory);
    }

    /// <summary>Available speaker names from the model.</summary>
    public IReadOnlyCollection<string> Speakers => _embeddings.GetAvailableSpeakers();

    /// <summary>The model variant this pipeline was created with.</summary>
    public QwenModelVariant ModelVariant => _variant;

    /// <summary>Maximum concurrent synthesis requests allowed for this pipeline instance.</summary>
    public int MaxConcurrency => _concurrencyGate.MaxConcurrency;

    /// <summary>
    /// Synthesizes speech from text and saves the output to a WAV file.
    /// </summary>
    /// <param name="text">Input text to synthesize. Must not be null, empty, and cannot exceed 10,000 characters.</param>
    /// <param name="speaker">Speaker name (must exist in model embeddings).</param>
    /// <param name="outputPath">Path where the output WAV file will be saved.</param>
    /// <param name="language">Language code (default: "auto" for auto-detection).</param>
    /// <param name="instruct">Optional instruction prompt for voice style modification.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <exception cref="ArgumentNullException">Thrown when text is null.</exception>
    /// <exception cref="ArgumentException">Thrown when text is empty or exceeds 10,000 characters.</exception>
    public async Task SynthesizeAsync(string text, string speaker, string outputPath, 
                                     string language = "auto", string? instruct = null,
                                     IProgress<string>? progress = null)
    {
        _ = await SynthesizeWithMetricsAsync(text, speaker, outputPath, language, instruct, progress);
    }

    /// <summary>
    /// Synthesizes speech from text, saves the output to a WAV file, and returns timing metrics.
    /// </summary>
    public async Task<TtsSynthesisMetrics> SynthesizeWithMetricsAsync(
        string text,
        string speaker,
        string outputPath,
        string language = "auto",
        string? instruct = null,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var artifacts = await SynthesizeToWaveformAsync(
            text,
            speaker,
            language,
            instruct,
            progress,
            cancellationToken);

        progress?.Report("Writing WAV file...");
        cancellationToken.ThrowIfCancellationRequested();
        await Task.Run(() => WavWriter.Write(outputPath, artifacts.Waveform, sampleRate: WavWriter.DefaultSampleRate), cancellationToken);

        var duration = artifacts.Waveform.Length / (double)WavWriter.DefaultSampleRate;
        progress?.Report(
            $"Saved {Path.GetFileName(outputPath)} ({artifacts.Waveform.Length} samples, {duration:F2}s) " +
            $"[queue {artifacts.Metrics.QueueLatency.TotalSeconds:F2}s, first audio {artifacts.Metrics.FirstAudioLatency.TotalSeconds:F2}s, total {artifacts.Metrics.TotalLatency.TotalSeconds:F2}s]");
        Console.WriteLine(
            $"Saved {outputPath} ({artifacts.Waveform.Length} samples, {duration:F2}s) " +
            $"[queue {artifacts.Metrics.QueueLatency.TotalSeconds:F2}s, first audio {artifacts.Metrics.FirstAudioLatency.TotalSeconds:F2}s, total {artifacts.Metrics.TotalLatency.TotalSeconds:F2}s]");

        return artifacts.Metrics;
    }

    /// <summary>
    /// Synthesizes speech and yields ordered WAV chunks whose byte concatenation reconstructs a valid WAV file.
    /// </summary>
    public async IAsyncEnumerable<TextToSpeechStreamingUpdate> SynthesizeStreamingAsync(
        string text,
        string speaker,
        string language = "auto",
        string? instruct = null,
        IProgress<string>? progress = null,
        int maxChunkBytes = 32 * 1024,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ValidateSynthesisText(text);

        yield return new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.SessionOpen,
            MediaType = StreamingMediaType,
            SampleRate = WavWriter.DefaultSampleRate,
            Channels = WavWriter.DefaultChannels,
            BitsPerSample = WavWriter.DefaultBitsPerSample,
            IsProgressive = false
        };

        var artifacts = await SynthesizeToWaveformAsync(
            text,
            speaker,
            language,
            instruct,
            progress,
            cancellationToken);

        foreach (var chunk in WavWriter.EnumerateWavChunks(
            artifacts.Waveform,
            sampleRate: WavWriter.DefaultSampleRate,
            channels: WavWriter.DefaultChannels,
            maxChunkBytes: maxChunkBytes))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return new TextToSpeechStreamingUpdate
            {
                Kind = TextToSpeechUpdateKind.AudioChunk,
                AudioData = chunk,
                MediaType = StreamingMediaType,
                SampleRate = WavWriter.DefaultSampleRate,
                Channels = WavWriter.DefaultChannels,
                BitsPerSample = WavWriter.DefaultBitsPerSample,
                IsProgressive = false
            };
        }

        yield return new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.SessionClose,
            MediaType = StreamingMediaType,
            SampleRate = WavWriter.DefaultSampleRate,
            Channels = WavWriter.DefaultChannels,
            BitsPerSample = WavWriter.DefaultBitsPerSample,
            IsProgressive = false,
            Metrics = artifacts.Metrics
        };
    }

    /// <summary>
    /// Convenience alias for <see cref="SynthesizeStreamingAsync"/> for callers that prefer a get-style name.
    /// </summary>
    public IAsyncEnumerable<TextToSpeechStreamingUpdate> GetStreamingAudioAsync(
        string text,
        string speaker,
        string language = "auto",
        string? instruct = null,
        IProgress<string>? progress = null,
        int maxChunkBytes = 32 * 1024,
        CancellationToken cancellationToken = default)
        => SynthesizeStreamingAsync(text, speaker, language, instruct, progress, maxChunkBytes, cancellationToken);

    private async Task<SynthesisArtifacts> SynthesizeToWaveformAsync(
        string text,
        string speaker,
        string language,
        string? instruct,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        ValidateSynthesisText(text);

        using var lease = await _concurrencyGate.EnterAsync(cancellationToken);
        var stopwatch = Stopwatch.StartNew();
        cancellationToken.ThrowIfCancellationRequested();

        // Variant-aware instruct handling: 0.6B does not support instruction control
        if (!string.IsNullOrEmpty(instruct) && !QwenModelVariantConfig.SupportsInstruct(_variant))
        {
            var warning = $"Warning: Instruction text ignored - {_variant} model does not support instruction control. Use 1.7B for style instructions.";
            progress?.Report(warning);
            Console.WriteLine(warning);
            instruct = null;
        }

        // Build prompt using tokenizer
        var tokenIds = _tokenizer.BuildCustomVoicePrompt(text, speaker, language, instruct);

        progress?.Report($"Tokenized input ({tokenIds.Length} tokens)");
        Console.WriteLine($"Generating speech ({tokenIds.Length} input tokens)...");

        // Generate audio codes via LM
        progress?.Report("Running language model inference...");
        var codes = _languageModel.Generate(tokenIds, speaker, language, cancellationToken: cancellationToken);
        
        int timesteps = codes.GetLength(2);
        progress?.Report($"Generated {timesteps} audio frames");
        Console.WriteLine($"Generated {timesteps} audio frames");

        // Decode to waveform via vocoder
        progress?.Report("Decoding waveform via vocoder...");
        var waveform = _vocoder.Decode(codes, cancellationToken);
        var firstAudioLatency = stopwatch.Elapsed;
        var metrics = new TtsSynthesisMetrics
        {
            QueueLatency = lease.QueueLatency,
            FirstAudioLatency = firstAudioLatency,
            TotalLatency = stopwatch.Elapsed,
            GeneratedFrames = timesteps,
            OutputSamples = waveform.Length
        };

        return new SynthesisArtifacts(waveform, metrics);
    }

    private static void ValidateSynthesisText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0)
            throw new ArgumentException("Text cannot be empty.", nameof(text));
        if (text.Length > 10000)
            throw new ArgumentException("Text exceeds maximum length of 10,000 characters.", nameof(text));
    }

    /// <summary>
    /// Synthesizes speech using a strongly-typed voice preset.
    /// </summary>
    /// <param name="text">Input text to synthesize. Must not be null, empty, and cannot exceed 10,000 characters.</param>
    /// <param name="speaker">Voice preset (enum) to use for synthesis.</param>
    /// <param name="outputPath">Path where the output WAV file will be saved.</param>
    /// <param name="language">Language code (default: "auto" for auto-detection).</param>
    /// <param name="instruct">Optional instruction prompt for voice style modification.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <exception cref="ArgumentNullException">Thrown when text is null.</exception>
    /// <exception cref="ArgumentException">Thrown when text is empty or exceeds 10,000 characters.</exception>
    public Task SynthesizeAsync(string text, QwenVoicePreset speaker, string outputPath,
                                string language = "auto", string? instruct = null,
                                IProgress<string>? progress = null)
        => SynthesizeAsync(text, speaker.ToSpeakerName(), outputPath, language, instruct, progress);

    /// <summary>
    /// Creates a TtsPipeline, automatically downloading model files if they are missing.
    /// </summary>
    /// <param name="modelDir">Directory to store/load model files. When null, uses the variant-specific default location.</param>
    /// <param name="repoId">HuggingFace repository ID. When null with a variant, uses the variant's default repo.</param>
    /// <param name="progress">Optional progress callback for download status.</param>
    /// <param name="sessionOptionsFactory">Optional factory for ONNX Runtime session options (e.g., for GPU acceleration).</param>
    /// <param name="vocoderSessionOptionsFactory">Optional separate factory for the vocoder model (e.g., CPU fallback for DirectML).</param>
    /// <param name="variant">Model size variant. Defaults to 0.6B for backward compatibility.</param>
    /// <param name="maxConcurrency">Maximum number of synthesis requests allowed to run concurrently per pipeline instance.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<TtsPipeline> CreateAsync(
        string? modelDir = null,
        string? repoId = null,
        IProgress<string>? progress = null,
        Func<SessionOptions>? sessionOptionsFactory = null,
        Func<SessionOptions>? vocoderSessionOptionsFactory = null,
        QwenModelVariant variant = QwenModelVariant.Qwen06B,
        int maxConcurrency = 1,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var (resolvedDir, resolvedRepo) = ModelDownloader.ResolveForVariant(variant, modelDir, repoId);

        if (!ModelDownloader.IsModelDownloaded(resolvedDir, variant))
        {
            progress?.Report("Model files not found — downloading from HuggingFace...");
            var downloadProgress = progress != null
                ? new Progress<ModelDownloadProgress>(p => progress.Report(p.Message))
                : null;
            await ModelDownloader.DownloadModelAsync(resolvedDir, resolvedRepo, downloadProgress, cancellationToken, variant);
            progress?.Report("Model download complete.");
        }
        cancellationToken.ThrowIfCancellationRequested();
        return new TtsPipeline(resolvedDir, sessionOptionsFactory, vocoderSessionOptionsFactory, variant, maxConcurrency);
    }

    /// <summary>
    /// Creates a TtsPipeline with detailed download progress reporting.
    /// </summary>
    /// <param name="modelDir">Directory to store/load model files. When null, uses the variant-specific default location.</param>
    /// <param name="downloadProgress">Detailed byte-level model download progress callback.</param>
    /// <param name="repoId">HuggingFace repository ID. When null with a variant, uses the variant's default repo.</param>
    /// <param name="sessionOptionsFactory">Optional factory for ONNX Runtime session options (e.g., for GPU acceleration).</param>
    /// <param name="vocoderSessionOptionsFactory">Optional separate factory for the vocoder model (e.g., CPU fallback for DirectML).</param>
    /// <param name="variant">Model size variant. Defaults to 0.6B for backward compatibility.</param>
    /// <param name="maxConcurrency">Maximum number of synthesis requests allowed to run concurrently per pipeline instance.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task<TtsPipeline> CreateAsync(
        string? modelDir,
        IProgress<ModelDownloadProgress> downloadProgress,
        string? repoId = null,
        Func<SessionOptions>? sessionOptionsFactory = null,
        Func<SessionOptions>? vocoderSessionOptionsFactory = null,
        QwenModelVariant variant = QwenModelVariant.Qwen06B,
        int maxConcurrency = 1,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var (resolvedDir, resolvedRepo) = ModelDownloader.ResolveForVariant(variant, modelDir, repoId);

        if (!ModelDownloader.IsModelDownloaded(resolvedDir, variant))
            await ModelDownloader.DownloadModelAsync(resolvedDir, resolvedRepo, downloadProgress, cancellationToken, variant);
        cancellationToken.ThrowIfCancellationRequested();
        return new TtsPipeline(resolvedDir, sessionOptionsFactory, vocoderSessionOptionsFactory, variant, maxConcurrency);
    }

    public void Dispose()
    {
        _concurrencyGate.Dispose();
        _tokenizer.Dispose();
        _embeddings.Dispose();
        _languageModel.Dispose();
        _vocoder.Dispose();
    }
}
