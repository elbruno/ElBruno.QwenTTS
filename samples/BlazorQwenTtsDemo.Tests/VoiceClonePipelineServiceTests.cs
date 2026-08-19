using BlazorQwenTtsDemo.Services;
using ElBruno.QwenTTS.Pipeline;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace BlazorQwenTtsDemo.Tests;

public sealed class VoiceClonePipelineServiceTests
{
    [Fact]
    public async Task GenerateAsync_UsesInjectedPipelineForSavedReferenceAndIclTranscript()
    {
        using var workspace = new TestWorkspace();
        var pipeline = new RecordingPipeline();
        var service = CreateService(workspace, pipeline);

        await service.InitializeAsync();
        var reference = await service.SaveReferenceAudioAsync([1, 2, 3, 4]);
        var result = await service.GenerateAsync(
            "Hello from the test",
            reference.FilePath,
            "Reference transcript",
            "english");

        Assert.True(service.IsReady);
        Assert.StartsWith("/generated/", result, StringComparison.Ordinal);
        Assert.EndsWith(".wav", result, StringComparison.Ordinal);
        Assert.Equal([1, 2, 3, 4], await File.ReadAllBytesAsync(reference.FilePath));
        Assert.Equal("Hello from the test", pipeline.Text);
        Assert.Equal(reference.FilePath, pipeline.ReferenceAudioPath);
        Assert.Equal("Reference transcript", pipeline.ReferenceTranscript);
        Assert.Equal("english", pipeline.Language);
        Assert.StartsWith(Path.Combine(workspace.WebRootPath, "generated"), pipeline.OutputPath,
            StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task GenerateAsync_DoesNotInvokePipelineWhenAlreadyCancelled()
    {
        using var workspace = new TestWorkspace();
        var pipeline = new RecordingPipeline();
        var service = CreateService(workspace, pipeline);
        await service.InitializeAsync();

        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            service.GenerateAsync("Hello", "reference.wav", null, "auto", cancellationToken: cancellation.Token));

        Assert.Null(pipeline.Text);
    }

    [Fact]
    public async Task GenerateAsync_PropagatesCancellationToRunningPipeline()
    {
        using var workspace = new TestWorkspace();
        var pipeline = new RecordingPipeline { WaitForCancellation = true };
        var service = CreateService(workspace, pipeline);
        await service.InitializeAsync();
        using var cancellation = new CancellationTokenSource();

        var generation = service.GenerateAsync(
            "Hello",
            "reference.wav",
            null,
            "auto",
            cancellationToken: cancellation.Token);
        await pipeline.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generation);
        Assert.True(pipeline.CancellationWasObserved);
    }

    [Fact]
    public async Task ExtractEmbeddingAsync_ThrowsWhenPipelineNotInitialized()
    {
        using var workspace = new TestWorkspace();
        var pipeline = new RecordingPipeline();
        var service = CreateService(workspace, pipeline);

        await Assert.ThrowsAsync<InvalidOperationException>(() => service.ExtractEmbeddingAsync("reference.wav"));
    }

    [Fact]
    public void Constructor_CreatesReferenceAndGeneratedDirectories()
    {
        using var workspace = new TestWorkspace();
        var pipeline = new RecordingPipeline();
        _ = CreateService(workspace, pipeline);

        Assert.True(Directory.Exists(Path.Combine(workspace.WebRootPath, "references")));
        Assert.True(Directory.Exists(Path.Combine(workspace.WebRootPath, "generated")));
    }

    private static VoiceClonePipelineService CreateService(TestWorkspace workspace, RecordingPipeline pipeline)
    {
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["VoiceClone:ModelDir"] = Path.Combine(workspace.RootPath, "models")
            })
            .Build();

        return new VoiceClonePipelineService(
            configuration,
            new TestWebHostEnvironment(workspace.RootPath, workspace.WebRootPath),
            NullLogger<VoiceClonePipelineService>.Instance,
            new RecordingPipelineFactory(pipeline));
    }

    private sealed class RecordingPipelineFactory(RecordingPipeline pipeline) : IVoiceClonePipelineFactory
    {
        public bool IsModelDownloaded(string modelDirectory) => true;

        public Task<IVoiceClonePipeline> CreateAsync(
            string modelDirectory,
            IProgress<ModelDownloadProgress> downloadProgress,
            CancellationToken cancellationToken) =>
            Task.FromResult<IVoiceClonePipeline>(pipeline);
    }

    private sealed class RecordingPipeline : IVoiceClonePipeline
    {
        public string? Text { get; private set; }
        public string? ReferenceAudioPath { get; private set; }
        public string? ReferenceTranscript { get; private set; }
        public string? Language { get; private set; }
        public string OutputPath { get; private set; } = string.Empty;
        public bool WaitForCancellation { get; init; }
        public bool CancellationWasObserved { get; private set; }
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public float[] ExtractSpeakerEmbedding(string referenceAudioPath) => [0.25f];

        public Task SynthesizeWithEmbeddingAsync(
            string text,
            float[] speakerEmbedding,
            string outputPath,
            string language = "auto",
            IProgress<string>? progress = null) => Task.CompletedTask;

        public async Task SynthesizeAsync(
            string text,
            string referenceAudioPath,
            string outputPath,
            string? referenceTranscript = null,
            string language = "auto",
            IProgress<string>? progress = null,
            CancellationToken cancellationToken = default)
        {
            Text = text;
            ReferenceAudioPath = referenceAudioPath;
            ReferenceTranscript = referenceTranscript;
            Language = language;
            OutputPath = outputPath;
            Started.TrySetResult();
            if (WaitForCancellation)
            {
                try
                {
                    await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    CancellationWasObserved = cancellationToken.IsCancellationRequested;
                    throw;
                }
            }
            progress?.Report("Fake generation complete.");
            await File.WriteAllBytesAsync(outputPath, [0]);
        }

        public void Dispose()
        {
        }
    }

    private sealed class TestWebHostEnvironment(string contentRootPath, string webRootPath) : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "BlazorQwenTtsDemo.Tests";
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
        public string WebRootPath { get; set; } = webRootPath;
        public string EnvironmentName { get; set; } = "Testing";
        public string ContentRootPath { get; set; } = contentRootPath;
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
    }

    private sealed class TestWorkspace : IDisposable
    {
        public TestWorkspace()
        {
            RootPath = Path.Combine(
                Environment.CurrentDirectory,
                "test-artifacts",
                nameof(VoiceClonePipelineServiceTests),
                Guid.NewGuid().ToString("N"));
            WebRootPath = Path.Combine(RootPath, "wwwroot");
            Directory.CreateDirectory(WebRootPath);
        }

        public string RootPath { get; }
        public string WebRootPath { get; }

        public void Dispose()
        {
            if (Directory.Exists(RootPath))
                Directory.Delete(RootPath, recursive: true);
        }
    }
}
