using ElBruno.QwenTTS.Pipeline;
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

namespace ElBruno.QwenTTS.Web.Services;

/// <summary>
/// Singleton service wrapping VoiceClonePipeline with thread-safe access.
/// Initializes asynchronously — Base models are downloaded automatically if missing.
/// </summary>
public sealed class VoiceClonePipelineService : IDisposable
{
    private VoiceClonePipeline? _pipeline;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly string _outputDir;
    private readonly string _referenceDir;
    private readonly string _modelDir;
    private bool _isInitializing;
    private bool _isReady;

    public bool IsReady => _isReady;
    public bool IsInitializing => _isInitializing;
    public bool IsModelDownloaded => VoiceCloningDownloader.IsModelDownloaded(_modelDir);

    public string ModelDirectory => _modelDir;

    public event Action<ModelDownloadProgress>? OnDownloadProgress;

    public VoiceClonePipelineService(IConfiguration config, IWebHostEnvironment env)
    {
        var modelDir = config["VoiceClone:ModelDir"];
        if (string.IsNullOrEmpty(modelDir))
        {
            _modelDir = VoiceCloningDownloader.DefaultModelDir;
        }
        else if (!Path.IsPathRooted(modelDir))
        {
            var solutionRoot = Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", ".."));
            _modelDir = Path.Combine(solutionRoot, modelDir);
        }
        else
        {
            _modelDir = modelDir;
        }

        _outputDir = Path.Combine(env.WebRootPath, "generated");
        _referenceDir = Path.Combine(env.WebRootPath, "references");
        Directory.CreateDirectory(_outputDir);
        Directory.CreateDirectory(_referenceDir);
    }

    /// <summary>
    /// Initialize the voice cloning pipeline, downloading Base models if needed.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_isReady || _isInitializing) return;
        _isInitializing = true;

        try
        {
            var progress = new Progress<ModelDownloadProgress>(p => OnDownloadProgress?.Invoke(p));
            _pipeline = await VoiceClonePipeline.CreateAsync(_modelDir, progress,
                cancellationToken: cancellationToken);
            _isReady = true;
        }
        finally
        {
            _isInitializing = false;
        }
    }

    /// <summary>
    /// Saves uploaded/recorded WAV bytes to disk and returns the file path and a playback URL.
    /// </summary>
    public async Task<(string FilePath, string Url)> SaveReferenceAudioAsync(byte[] wavBytes)
    {
        var fileName = $"ref_{Guid.NewGuid():N}.wav";
        var filePath = Path.Combine(_referenceDir, fileName);
        await File.WriteAllBytesAsync(filePath, wavBytes);
        return (filePath, $"/references/{fileName}");
    }

    /// <summary>
    /// Extracts a 1024-dim speaker embedding from a reference WAV file.
    /// </summary>
    public async Task<float[]> ExtractEmbeddingAsync(string wavPath)
    {
        if (!_isReady)
            throw new InvalidOperationException("Pipeline not initialized. Call InitializeAsync first.");

        await _semaphore.WaitAsync();
        try
        {
            return await Task.Run(() => _pipeline!.ExtractSpeakerEmbedding(wavPath));
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Generates speech with a cloned voice and returns the relative URL path.
    /// </summary>
    public async Task<string> GenerateAsync(string text, float[] embedding, string language,
                                            IProgress<string>? progress = null)
    {
        if (!_isReady)
            throw new InvalidOperationException("Pipeline not initialized. Call InitializeAsync first.");

        var fileName = $"{Guid.NewGuid():N}.wav";
        var filePath = Path.Combine(_outputDir, fileName);

        await _semaphore.WaitAsync();
        try
        {
            await Task.Run(async () =>
                await _pipeline!.SynthesizeWithEmbeddingAsync(text, embedding, filePath, language, progress));
        }
        finally
        {
            _semaphore.Release();
        }

        return $"/generated/{fileName}";
    }

    public void Dispose()
    {
        _semaphore.Dispose();
        _pipeline?.Dispose();
    }
}
