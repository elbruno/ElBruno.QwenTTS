using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

public class ModelDownloaderTests : IDisposable
{
    private readonly string _tempDir;

    public ModelDownloaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void IsModelReady_EmptyDir_ReturnsFalse()
    {
        Assert.False(ModelDownloader.IsModelReady(_tempDir));
    }

    [Fact]
    public void IsModelReady_NonExistentDir_ReturnsFalse()
    {
        Assert.False(ModelDownloader.IsModelReady(Path.Combine(_tempDir, "nonexistent")));
    }

    [Fact]
    public void GetMissingFiles_EmptyDir_ReturnsAllExpectedFiles()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        // Should include ONNX models, embeddings, tokenizer files
        Assert.Contains(missing, f => f == "talker_prefill.onnx");
        Assert.Contains(missing, f => f == "vocoder.onnx");
        Assert.Contains(missing, f => f == "embeddings/config.json");
        Assert.Contains(missing, f => f == "tokenizer/vocab.json");
        Assert.Contains(missing, f => f == "embeddings/cp_codec_embedding_0.npy");
        Assert.True(missing.Count >= 33); // 7 onnx + 8 embeddings + 15 cp + 2 tokenizer + 1 speaker_ids
    }

    [Fact]
    public void GetMissingFiles_WithSomeFiles_ReturnsOnlyMissing()
    {
        // Create a few expected files
        Directory.CreateDirectory(Path.Combine(_tempDir, "tokenizer"));
        File.WriteAllText(Path.Combine(_tempDir, "tokenizer", "vocab.json"), "{}");
        File.WriteAllText(Path.Combine(_tempDir, "tokenizer", "merges.txt"), "");

        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        Assert.DoesNotContain(missing, f => f == "tokenizer/vocab.json");
        Assert.DoesNotContain(missing, f => f == "tokenizer/merges.txt");
        Assert.Contains(missing, f => f == "talker_prefill.onnx");
    }

    [Fact]
    public void IsModelReady_AllFilesPresent_ReturnsTrue()
    {
        // Create all expected files as empty stubs
        var missing = ModelDownloader.GetMissingFiles(_tempDir);
        foreach (var file in missing)
        {
            var path = Path.Combine(_tempDir, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, "stub");
        }

        Assert.True(ModelDownloader.IsModelReady(_tempDir));
        Assert.Empty(ModelDownloader.GetMissingFiles(_tempDir));
    }

    [Fact]
    public async Task DownloadModelAsync_AlreadyReady_SkipsDownload()
    {
        // Create all files as stubs
        var missing = ModelDownloader.GetMissingFiles(_tempDir);
        foreach (var file in missing)
        {
            var path = Path.Combine(_tempDir, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, "stub");
        }

        string? lastMessage = null;
        var progress = new Progress<ModelDownloadProgress>(p => lastMessage = p.Message);

        await ModelDownloader.DownloadModelAsync(_tempDir, progress: progress);

        // Small delay for progress callback
        await Task.Delay(100);
        Assert.Contains("already present", lastMessage ?? "");
    }

    [Fact]
    public async Task DownloadModelAsync_Cancellation_ThrowsOperationCanceled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            ModelDownloader.DownloadModelAsync(_tempDir, cancellationToken: cts.Token));
    }

    [Fact]
    public void DefaultRepoId_IsCorrect()
    {
        Assert.Equal("elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX", ModelDownloader.DefaultRepoId);
    }

    [Fact]
    public void ModelDownloadProgress_Percentage_Calculation()
    {
        var progress = new ModelDownloadProgress(5, 10, "test.onnx", "Downloading...", 512_000, 1_024_000);
        Assert.Equal(50.0, progress.FilePercentage);
        Assert.Equal(50.0, progress.BytePercentage);

        var empty = new ModelDownloadProgress(0, 0, null, "Done", 0, 0);
        Assert.Equal(0.0, empty.FilePercentage);
        Assert.Equal(0.0, empty.BytePercentage);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
