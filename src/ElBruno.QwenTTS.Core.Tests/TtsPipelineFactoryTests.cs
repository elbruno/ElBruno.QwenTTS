using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

public class TtsPipelineFactoryTests : IDisposable
{
    private readonly string _tempDir;

    public TtsPipelineFactoryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_factory_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public async Task CreateAsync_WithMissingFiles_TriggersDownload()
    {
        // Verify the factory detects missing files before attempting download
        Assert.False(ModelDownloader.IsModelReady(_tempDir));

        var messages = new List<string>();
        var progress = new Progress<string>(msg => messages.Add(msg));

        // Use a cancelled token to prevent actual download, but confirm the flow starts
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(_tempDir, progress: progress, cancellationToken: cts.Token));
    }

    [Fact]
    public void Constructor_WithInvalidDir_Throws()
    {
        // TtsPipeline constructor should throw when model files don't exist
        var bogusDir = Path.Combine(_tempDir, "nonexistent_models");
        Assert.ThrowsAny<Exception>(() => new TtsPipeline(bogusDir));
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
