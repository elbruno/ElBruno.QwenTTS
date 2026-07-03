using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// A production-ready <see cref="ITextToSpeechClient"/> backed by QwenTTS.
/// Provides thread-safe lazy initialization, in-memory synthesis, streaming
/// support, and proper resource disposal.
/// </summary>
/// <remarks>
/// The underlying <see cref="TtsPipeline"/> is lazily initialized on first use
/// and then reused across requests with bounded concurrency enforced by the pipeline instance.
/// Models are automatically downloaded from HuggingFace if not present.
/// </remarks>
public sealed class QwenTextToSpeechClient : ITextToSpeechClient, Microsoft.Extensions.AI.ITextToSpeechClient
{
    private static readonly Uri ProviderUri = new("https://github.com/elbruno/ElBruno.QwenTTS");
    private static readonly TextToSpeechClientMetadata ClientMetadata = new(
        providerName: "ElBruno.QwenTTS",
        providerUri: ProviderUri,
        defaultModelId: "qwen3-tts");

    private readonly string _defaultVoice;
    private readonly string _defaultLanguage;
    private readonly string? _defaultInstruct;
    private readonly string? _modelDir;
    private readonly string? _repoId;
    private readonly QwenModelVariant _variant;
    private readonly ExecutionProvider _executionProvider;
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
    /// <param name="defaultInstruct">Optional default instruction text applied when a request does not provide one.</param>
    /// <param name="modelDir">Optional model directory. When null, uses the variant-specific default.</param>
    /// <param name="repoId">HuggingFace repository ID. When null, uses the variant's default repo.</param>
    /// <param name="variant">Model size variant. Defaults to 0.6B.</param>
    /// <param name="executionProvider">Execution provider metadata reported for Microsoft.Extensions.AI responses.</param>
    /// <param name="sessionOptionsFactory">Optional ONNX Runtime session options factory (e.g., for GPU).</param>
    /// <param name="vocoderSessionOptionsFactory">Optional separate factory for the vocoder model.</param>
    /// <param name="maxConcurrency">Maximum concurrent synthesis requests allowed for the shared pipeline instance.</param>
    public QwenTextToSpeechClient(
        string defaultVoice = "ryan",
        string defaultLanguage = "auto",
        string? defaultInstruct = null,
        string? modelDir = null,
        string? repoId = null,
        QwenModelVariant variant = QwenModelVariant.Qwen06B,
        ExecutionProvider executionProvider = ExecutionProvider.Cpu,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? sessionOptionsFactory = null,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? vocoderSessionOptionsFactory = null,
        int maxConcurrency = 1)
        : this(
            defaultVoice,
            defaultLanguage,
            defaultInstruct,
            modelDir,
            repoId,
            variant,
            executionProvider,
            sessionOptionsFactory,
            vocoderSessionOptionsFactory,
            maxConcurrency,
            pipelineFactory: null)
    {
    }

    internal QwenTextToSpeechClient(
        string defaultVoice,
        string defaultLanguage,
        string? defaultInstruct,
        string? modelDir,
        string? repoId,
        QwenModelVariant variant,
        ExecutionProvider executionProvider,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? sessionOptionsFactory,
        Func<Microsoft.ML.OnnxRuntime.SessionOptions>? vocoderSessionOptionsFactory,
        int maxConcurrency,
        Func<CancellationToken, Task<ITtsPipeline>>? pipelineFactory)
    {
        _defaultVoice = defaultVoice;
        _defaultLanguage = defaultLanguage;
        _defaultInstruct = defaultInstruct;
        _modelDir = modelDir;
        _repoId = repoId;
        _variant = variant;
        _executionProvider = executionProvider;
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
        var instruct = options?.Instruct ?? _defaultInstruct;
        cancellationToken.ThrowIfCancellationRequested();

        var audio = await _pipeline!.SynthesizeToPcmAsync(
            text,
            voice,
            language,
            instruct,
            cancellationToken: cancellationToken);

        return new TextToSpeechResponse
        {
            AudioData = audio.ToWavBytes().ToArray(),
            MediaType = "audio/wav",
            SampleRate = audio.SampleRate,
            ModelId = options?.ModelId ?? "qwen3-tts",
            Metrics = audio.Metrics,
        };
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
        var instruct = options?.Instruct ?? _defaultInstruct;

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

    /// <inheritdoc />
    public async Task<Microsoft.Extensions.AI.TextToSpeechResponse> GetAudioAsync(
        string text,
        Microsoft.Extensions.AI.TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var requestOptions = ConvertOptions(options);
        var response = await SynthesizeToMemoryAsync(text, requestOptions, cancellationToken);
        return CreateAiResponse(response, requestOptions);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<Microsoft.Extensions.AI.TextToSpeechResponseUpdate> GetStreamingAudioAsync(
        string text,
        Microsoft.Extensions.AI.TextToSpeechOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var requestOptions = ConvertOptions(options);

        await foreach (var update in SynthesizeStreamingAsync(text, requestOptions, cancellationToken)
            .WithCancellation(cancellationToken))
        {
            yield return CreateAiUpdate(update, requestOptions);
        }
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(TextToSpeechClientMetadata))
        {
            return ClientMetadata;
        }

        if (serviceType == typeof(ITtsPipeline))
        {
            return _pipeline;
        }

        if (serviceType == typeof(QwenTextToSpeechClient) ||
            serviceType == typeof(ITextToSpeechClient) ||
            serviceType == typeof(Microsoft.Extensions.AI.ITextToSpeechClient))
        {
            return this;
        }

        return null;
    }

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

    private TextToSpeechOptions ConvertOptions(Microsoft.Extensions.AI.TextToSpeechOptions? options)
    {
        ValidateOptions(options);

        return new TextToSpeechOptions
        {
            VoiceId = options?.VoiceId,
            Language = options?.Language,
            Instruct = TryGetString(options?.AdditionalProperties, QwenTextToSpeechMetadataKeys.Instruct),
            ModelId = options?.ModelId
        };
    }

    private static void ValidateOptions(Microsoft.Extensions.AI.TextToSpeechOptions? options)
    {
        if (options is null)
        {
            return;
        }

        if (!string.IsNullOrWhiteSpace(options.AudioFormat) &&
            !options.AudioFormat.Equals("audio/wav", StringComparison.OrdinalIgnoreCase) &&
            !options.AudioFormat.Equals("audio/x-wav", StringComparison.OrdinalIgnoreCase) &&
            !options.AudioFormat.Equals("wav", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException("QwenTextToSpeechClient only supports WAV output via audio/wav.");
        }

        if (options.Speed is not null || options.Pitch is not null || options.Volume is not null)
        {
            throw new NotSupportedException("QwenTextToSpeechClient does not support speed, pitch, or volume controls.");
        }

        if (options.AdditionalProperties?.ContainsKey(QwenTextToSpeechMetadataKeys.VoiceCloning) == true)
        {
            throw new NotSupportedException(
                "QwenTextToSpeechClient does not support voice-cloning input through Microsoft.Extensions.AI. Use ElBruno.QwenTTS.VoiceCloning for reference-audio synthesis.");
        }
    }

    private Microsoft.Extensions.AI.TextToSpeechResponse CreateAiResponse(
        TextToSpeechResponse response,
        TextToSpeechOptions requestOptions)
    {
        return new Microsoft.Extensions.AI.TextToSpeechResponse(
            [new DataContent(response.AudioData, response.MediaType)])
        {
            AdditionalProperties = CreateAdditionalProperties(requestOptions),
            ModelId = response.ModelId,
            RawRepresentation = response
        };
    }

    private Microsoft.Extensions.AI.TextToSpeechResponseUpdate CreateAiUpdate(
        TextToSpeechStreamingUpdate update,
        TextToSpeechOptions requestOptions)
    {
        IList<AIContent> contents = update.AudioData is null
            ? new List<AIContent>()
            : [new DataContent(update.AudioData, update.MediaType ?? "audio/wav")];

        return new Microsoft.Extensions.AI.TextToSpeechResponseUpdate(contents)
        {
            AdditionalProperties = CreateAdditionalProperties(requestOptions),
            Kind = update.Kind switch
            {
                TextToSpeechUpdateKind.SessionOpen => Microsoft.Extensions.AI.TextToSpeechResponseUpdateKind.SessionOpen,
                TextToSpeechUpdateKind.AudioChunk => Microsoft.Extensions.AI.TextToSpeechResponseUpdateKind.AudioUpdated,
                TextToSpeechUpdateKind.SessionClose => Microsoft.Extensions.AI.TextToSpeechResponseUpdateKind.SessionClose,
                _ => Microsoft.Extensions.AI.TextToSpeechResponseUpdateKind.Error
            },
            ModelId = requestOptions.ModelId ?? "qwen3-tts",
            RawRepresentation = update
        };
    }

    private AdditionalPropertiesDictionary CreateAdditionalProperties(TextToSpeechOptions requestOptions)
    {
        var properties = new AdditionalPropertiesDictionary
        {
            [QwenTextToSpeechMetadataKeys.Variant] = QwenModelVariantConfig.GetModelSubDir(_variant),
            [QwenTextToSpeechMetadataKeys.Speaker] = requestOptions.VoiceId ?? _defaultVoice,
            [QwenTextToSpeechMetadataKeys.Language] = requestOptions.Language ?? _defaultLanguage,
            [QwenTextToSpeechMetadataKeys.VoiceCloning] = false,
            [QwenTextToSpeechMetadataKeys.ExecutionProvider] = _executionProvider.ToString()
        };

        var instruct = requestOptions.Instruct ?? _defaultInstruct;
        if (instruct is not null)
        {
            properties[QwenTextToSpeechMetadataKeys.Instruct] = instruct;
        }

        return properties;
    }

    private static string? TryGetString(AdditionalPropertiesDictionary? properties, string key)
    {
        if (properties is null || !properties.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        if (value is string text)
        {
            return text;
        }

        throw new NotSupportedException($"Additional property '{key}' must be a string.");
    }

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
