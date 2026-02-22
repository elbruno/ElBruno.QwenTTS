namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Abstraction for text-to-speech synthesis pipelines.
/// Enables dependency injection, mocking, and testability.
/// </summary>
public interface ITtsPipeline : IDisposable
{
    /// <summary>Available speaker names from the model.</summary>
    IReadOnlyCollection<string> Speakers { get; }

    /// <summary>
    /// Synthesizes speech from text and saves to a WAV file.
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="speaker">Speaker name (e.g., "ryan") or <see cref="QwenVoicePreset"/> string value.</param>
    /// <param name="outputPath">Output WAV file path.</param>
    /// <param name="language">Language code ("auto", "english", "chinese", etc.).</param>
    /// <param name="instruct">Optional instruction for speech style.</param>
    /// <param name="progress">Optional progress callback.</param>
    Task SynthesizeAsync(string text, string speaker, string outputPath,
                         string language = "auto", string? instruct = null,
                         IProgress<string>? progress = null);
}
