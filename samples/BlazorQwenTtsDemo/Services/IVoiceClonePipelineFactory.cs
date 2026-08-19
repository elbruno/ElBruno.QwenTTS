using ElBruno.QwenTTS.Pipeline;
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

namespace BlazorQwenTtsDemo.Services;

/// <summary>
/// Creates voice-cloning pipeline adapters for the sample host.
/// </summary>
public interface IVoiceClonePipelineFactory
{
    bool IsModelDownloaded(string modelDirectory);

    Task<IVoiceClonePipeline> CreateAsync(
        string modelDirectory,
        IProgress<ModelDownloadProgress> downloadProgress,
        CancellationToken cancellationToken);
}

/// <summary>
/// Represents the voice-cloning operations used by the sample host.
/// </summary>
public interface IVoiceClonePipeline : IDisposable
{
    float[] ExtractSpeakerEmbedding(string referenceAudioPath);

    Task SynthesizeWithEmbeddingAsync(
        string text,
        float[] speakerEmbedding,
        string outputPath,
        string language = "auto",
        IProgress<string>? progress = null);

    Task SynthesizeAsync(
        string text,
        string referenceAudioPath,
        string outputPath,
        string? referenceTranscript = null,
        string language = "auto",
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default);
}

internal sealed class VoiceClonePipelineFactory : IVoiceClonePipelineFactory
{
    public bool IsModelDownloaded(string modelDirectory) =>
        VoiceCloningDownloader.IsModelDownloaded(modelDirectory);

    public async Task<IVoiceClonePipeline> CreateAsync(
        string modelDirectory,
        IProgress<ModelDownloadProgress> downloadProgress,
        CancellationToken cancellationToken)
    {
        var pipeline = await VoiceClonePipeline.CreateAsync(
            modelDirectory,
            downloadProgress,
            cancellationToken: cancellationToken);
        return new VoiceClonePipelineAdapter(pipeline);
    }
}

internal sealed class VoiceClonePipelineAdapter(VoiceClonePipeline pipeline) : IVoiceClonePipeline
{
    public float[] ExtractSpeakerEmbedding(string referenceAudioPath) =>
        pipeline.ExtractSpeakerEmbedding(referenceAudioPath);

    public Task SynthesizeWithEmbeddingAsync(
        string text,
        float[] speakerEmbedding,
        string outputPath,
        string language = "auto",
        IProgress<string>? progress = null) =>
        pipeline.SynthesizeWithEmbeddingAsync(text, speakerEmbedding, outputPath, language, progress);

    public Task SynthesizeAsync(
        string text,
        string referenceAudioPath,
        string outputPath,
        string? referenceTranscript = null,
        string language = "auto",
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default) =>
        pipeline.SynthesizeAsync(
            text,
            referenceAudioPath,
            outputPath,
            referenceTranscript,
            language,
            progress,
            cancellationToken);

    public void Dispose() => pipeline.Dispose();
}
