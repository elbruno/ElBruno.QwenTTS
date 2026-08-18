using ElBruno.QwenTTS.Pipeline;
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

namespace BlazorQwenTtsDemo.Services;

public interface IModelDownloadAdapter
{
    string DisplayName { get; }
    bool IsDownloaded();
    Task DownloadAsync(IProgress<ModelDownloadProgress> progress, CancellationToken cancellationToken = default);
}

public sealed class BaseTtsModelDownloadAdapter : IModelDownloadAdapter
{
    public string DisplayName => "Base TTS models";

    public bool IsDownloaded() => ModelDownloader.IsModelDownloaded();

    public Task DownloadAsync(IProgress<ModelDownloadProgress> progress, CancellationToken cancellationToken = default) =>
        ModelDownloader.DownloadModelAsync(progress: progress, cancellationToken: cancellationToken);
}

public sealed class VoiceCloningModelDownloadAdapter : IModelDownloadAdapter
{
    public string DisplayName => "Voice-cloning models";

    public bool IsDownloaded() => VoiceCloningDownloader.IsModelDownloaded();

    public Task DownloadAsync(IProgress<ModelDownloadProgress> progress, CancellationToken cancellationToken = default) =>
        VoiceCloningDownloader.DownloadModelAsync(progress: progress, cancellationToken: cancellationToken);
}

public sealed record ModelDownloadState(
    string DisplayName,
    bool IsDownloaded,
    bool IsDownloading,
    ModelDownloadProgress? Progress,
    string? Error);

public class ModelDownloadController
{
    private readonly IModelDownloadAdapter _adapter;
    private readonly SemaphoreSlim _downloadGate = new(1, 1);

    public ModelDownloadController(IModelDownloadAdapter adapter)
    {
        _adapter = adapter;
        State = new ModelDownloadState(adapter.DisplayName, false, false, null, null);
    }

    public ModelDownloadState State { get; private set; }

    public event Action? Changed;

    public void Refresh()
    {
        State = State with { IsDownloaded = _adapter.IsDownloaded() };
        NotifyChanged();
    }

    public async Task DownloadAsync()
    {
        if (!await _downloadGate.WaitAsync(0))
            return;

        try
        {
            Refresh();
            if (State.IsDownloaded)
                return;

            State = State with { IsDownloading = true, Error = null, Progress = null };
            NotifyChanged();

            await _adapter.DownloadAsync(
                new CallbackProgress(progress =>
                {
                    State = State with { Progress = progress };
                    NotifyChanged();
                }));

            State = State with { IsDownloaded = _adapter.IsDownloaded(), Progress = null };
        }
        catch
        {
            State = State with
            {
                IsDownloaded = _adapter.IsDownloaded(),
                Error = "The download failed. Please retry.",
                Progress = null
            };
        }
        finally
        {
            State = State with { IsDownloading = false };
            NotifyChanged();
            _downloadGate.Release();
        }
    }

    private void NotifyChanged() => Changed?.Invoke();

    private sealed class CallbackProgress(Action<ModelDownloadProgress> report) : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value) => report(value);
    }
}

public sealed class BaseTtsModelDownloadController : ModelDownloadController
{
    public BaseTtsModelDownloadController(BaseTtsModelDownloadAdapter adapter) : base(adapter) { }
}

public sealed class VoiceCloningModelDownloadController : ModelDownloadController
{
    public VoiceCloningModelDownloadController(VoiceCloningModelDownloadAdapter adapter) : base(adapter) { }
}
