using System.Runtime.CompilerServices;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// A production-ready <see cref="ITextToSpeechClient"/> backed by QwenTTS.
/// Provides thread-safe lazy initialization, in-memory synthesis (with automatic
/// temp file cleanup), streaming support, and proper resource disposal.
/// </summary>
/// <remarks>
/// The underlying <see cref="TtsPipeline"/> is lazily initialized on first use
/// and then reused across requests with bounded concurrency enforced by the pipeline instance.
/// Models are automatically downloaded from HuggingFace if not present.
/// </remarks>
public sealed class QwenTextToSpeechClient : ITextToSpeechClient
{
    private readonly string _defaultVoice;
    private readonly string _defaultLanguage;
    private readonly string? _modelDir;
    private readonly string? _repoId;
    private readonly QwenModelVariant _variant;
    private readonly Func<Microsoft.ML.OnnxRuntime.SessionOptions>? _sessionOptionsFactory;
    private readonly Func<Microsoft.ML.OnnxRuntime.SessionOptions>? _vocoderSessionOptionsFactory;
    private readonly int _maxConcurrency;
    private readonly Func<CancellationToken, Task<ITtsPipeline>> _pipelineFactory;

    private ITtsPipeline? _pipeline;
    private bool _disposed;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    /// <summary>
    /// Creates a new <see cref="QwenTextToSpeechClient"/>.
    /// </summary>
    /// <param name="defaultVoice">Default speaker name (e.g., "ryan", "serena"). Default: "ryan".</param>
    /// <param name="defaultLanguage">Default language (e.g., "english", "auto"). Default: "auto".</param>
    /// <param name="modelDir">Optional model directory. When null, uses the variant-specific default.</param>
    /// <param name="repoId">HuggingFace repository ID. When null, uses the variant's default repo.</param>
    /// <param name="variant">Model size variant. Defaults to 0.6B.</param>
    /// <param name="sessionOptionsFactory">Optional ONNX Runtime session options factory (e.g., for GPU).</param>
    /// <param name="vocoderSessionOptionsFactory">Optional separate factory for the vocoder model.</param>
    /// <param name="maxConcurrency">Maximum concurrent synthesis requests allowed for the shared pipeline instance.</param>
    public QwenTextToSpeechClient(
        string defaultVoice = "ryan",
        string defaultLanguage = "auto",
        string? modelDir = null,
        string? repoId = null,
        QwenModelVariant variant = QwenModelVariant.Qwen06B,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? sessionOptionsFactory = null,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? vocoderSessionOptionsFactory = null,
        int maxConcurrency = 1)
        : this(
            defaultVoice,
            defaultLanguage,
            modelDir,
            repoId,
            variant,
            sessionOptionsFactory,
            vocoderSessionOptionsFactory,
            maxConcurrency,
            pipelineFactory: null)
    {
    }

    internal QwenTextToSpeechClient(
        string defaultVoice,
        string defaultLanguage,
        string? modelDir,
        string? repoId,
        QwenModelVariant variant,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? sessionOptionsFactory,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? vocoderSessionOptionsFactory,
        int maxConcurrency,
        Func<CancellationToken, Task<ITtsPipeline>>? pipelineFactory)
    {
        _defaultVoice = defaultVoice;
        _defaultLanguage = defaultLanguage;
        _modelDir = modelDir;
        _repoId = repoId;
        _variant = variant;
        _sessionOptionsFactory = sessionOptionsFactory;
        _vocoderSessionOptionsFactory = vocoderSessionOptionsFactory;
        _maxConcurrency = maxConcurrency;
        _pipelineFactory = pipelineFactory ?? CreatePipelineAsync;
    }

    /// <inheritdoc />
    public async Task<TextToSpeechResponse> SynthesizeToMemoryAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(text);

        await EnsureInitializedAsync(cancellationToken);

        var voice = options?.VoiceId ?? _defaultVoice;
        var language = options?.Language ?? _defaultLanguage;
        var instruct = options?.Instruct;
        cancellationToken.ThrowIfCancellationRequested();

        // Write to temp file, read into memory, then clean up
        var tempPath = Path.Combine(Path.GetTempPath(), $"qwentts_{Guid.NewGuid():N}.wav");
        try
        {
            var metrics = await _pipeline!.SynthesizeWithMetricsAsync(
                text, voice, tempPath, language, instruct, cancellationToken: cancellationToken);
            var audioData = await File.ReadAllBytesAsync(tempPath, cancellationToken);

            return new TextToSpeechResponse
            {
                AudioData = audioData,
                MediaType = "audio/wav",
                SampleRate = ElBruno.QwenTTS.Audio.WavWriter.DefaultSampleRate,
                ModelId = options?.ModelId ?? "qwen3-tts",
                Metrics = metrics,
            };
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* cleanup best-effort */ }
        }
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<TextToSpeechStreamingUpdate> SynthesizeStreamingAsync(
        string text,
        TextToSpeechOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(text);

        await EnsureInitializedAsync(cancellationToken);

        var voice = options?.VoiceId ?? _defaultVoice;
        var language = options?.Language ?? _defaultLanguage;
        var instruct = options?.Instruct;

        await foreach (var update in _pipeline!.SynthesizeStreamingAsync(
            text,
            voice,
            language,
            instruct,
            cancellationToken: cancellationToken).WithCancellation(cancellationToken))
        {
            yield return update;
        }
    }

    /// <summary>
    /// Convenience alias for <see cref="SynthesizeStreamingAsync"/> for callers that prefer a get-style name.
    /// </summary>
    public IAsyncEnumerable<TextToSpeechStreamingUpdate> GetStreamingAudioAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default)
        => SynthesizeStreamingAsync(text, options, cancellationToken);

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_pipeline is not null) return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_pipeline is not null) return;
            _pipeline = await _pipelineFactory(cancellationToken);
        }
        finally
        {
            _initLock.Release();
        }
    }

    private async Task<ITtsPipeline> CreatePipelineAsync(CancellationToken cancellationToken)
        => await TtsPipeline.CreateAsync(
            _modelDir,
            repoId: _repoId,
            variant: _variant,
            sessionOptionsFactory: _sessionOptionsFactory,
            vocoderSessionOptionsFactory: _vocoderSessionOptionsFactory,
            maxConcurrency: _maxConcurrency,
            cancellationToken: cancellationToken);

    /// <summary>Disposes the underlying pipeline, semaphore, and any held resources.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _pipeline?.Dispose();
        _pipeline = null;
        _initLock.Dispose();

        GC.SuppressFinalize(this);
    }
}
