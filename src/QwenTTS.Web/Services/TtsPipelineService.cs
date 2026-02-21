using QwenTTS.Pipeline;

namespace QwenTTS.Web.Services;

/// <summary>
/// Singleton service that wraps TtsPipeline with thread-safe access.
/// </summary>
public sealed class TtsPipelineService : IDisposable
{
    private readonly TtsPipeline _pipeline;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly string _outputDir;

    public TtsPipelineService(IConfiguration config, IWebHostEnvironment env)
    {
        var modelDir = config["TTS:ModelDir"] ?? "python/onnx_runtime";
        // Resolve relative paths from the solution root (two levels up from web project)
        if (!Path.IsPathRooted(modelDir))
        {
            var solutionRoot = Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", ".."));
            modelDir = Path.Combine(solutionRoot, modelDir);
        }
        _pipeline = new TtsPipeline(modelDir);
        _outputDir = Path.Combine(env.WebRootPath, "generated");
        Directory.CreateDirectory(_outputDir);
    }

    public IReadOnlyCollection<string> Speakers => _pipeline.Speakers;

    /// <summary>
    /// Generates a WAV file and returns the relative URL path.
    /// </summary>
    public async Task<string> GenerateAsync(string text, string speaker, string language, string? instruct)
    {
        var fileName = $"{Guid.NewGuid():N}.wav";
        var filePath = Path.Combine(_outputDir, fileName);

        await _semaphore.WaitAsync();
        try
        {
            await _pipeline.SynthesizeAsync(text, speaker, filePath, language, instruct);
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
        _pipeline.Dispose();
    }
}
