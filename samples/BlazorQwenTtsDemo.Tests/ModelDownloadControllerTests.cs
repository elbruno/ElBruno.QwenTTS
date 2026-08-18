using System.Diagnostics;
using BlazorQwenTtsDemo.Services;
using ElBruno.QwenTTS.Pipeline;

namespace BlazorQwenTtsDemo.Tests;

public class ModelDownloadControllerTests
{
    [Fact]
    public async Task DownloadAsync_UsesHostAdapterAndUpdatesTheStatus()
    {
        var adapter = new FakeAdapter();
        var controller = new ModelDownloadController(adapter);
        var snapshots = new List<ModelDownloadState>();
        controller.Changed += () => snapshots.Add(controller.State);

        controller.Refresh();
        await controller.DownloadAsync();

        Assert.Equal(1, adapter.DownloadCalls);
        Assert.True(controller.State.IsDownloaded);
        Assert.False(controller.State.IsDownloading);
        Assert.Null(controller.State.Error);
        Assert.Contains(snapshots, state => state.Progress is not null);
    }

    [Fact]
    public void TelemetryOperations_DoNotExposeUserContentTags()
    {
        Activity? captured = null;
        using var listener = new ActivityListener
        {
            ShouldListenTo = source => source.Name == "ElBruno.QwenTTS",
            Sample = static (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
            ActivityStarted = activity => captured = activity
        };
        ActivitySource.AddActivityListener(listener);

        using var activity = QwenTtsTelemetry.StartVoiceCloning("synthesize");

        Assert.NotNull(captured);
        Assert.Equal("voice_clone.synthesize", captured!.GetTagItem("gen_ai.operation.name"));
        Assert.DoesNotContain(captured.TagObjects, tag =>
            tag.Key.Contains("prompt", StringComparison.OrdinalIgnoreCase) ||
            tag.Key.Contains("reference", StringComparison.OrdinalIgnoreCase) ||
            tag.Key.Contains("embedding", StringComparison.OrdinalIgnoreCase) ||
            tag.Key.Contains("path", StringComparison.OrdinalIgnoreCase));
    }

    private sealed class FakeAdapter : IModelDownloadAdapter
    {
        public string DisplayName => "Test models";
        public int DownloadCalls { get; private set; }
        private bool Downloaded { get; set; }

        public bool IsDownloaded() => Downloaded;

        public Task DownloadAsync(IProgress<ModelDownloadProgress> progress, CancellationToken cancellationToken = default)
        {
            DownloadCalls++;
            progress.Report(new ModelDownloadProgress(1, 1, "model.onnx", "Downloading", 100, 100));
            Downloaded = true;
            return Task.CompletedTask;
        }
    }
}
