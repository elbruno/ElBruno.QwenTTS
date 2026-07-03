namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// A high-level text-to-speech client aligned with Microsoft.Extensions.AI patterns.
/// Provides thread-safe lazy initialization, in-memory synthesis, and streaming support.
/// </summary>
public interface ITextToSpeechClient : IDisposable
{
    /// <summary>
    /// Synthesizes speech from text and returns audio data in memory.
    /// The pipeline is lazily initialized on first call (thread-safe).
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="options">Optional synthesis options (voice, language, etc.).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A response containing audio data and metadata.</returns>
    Task<TextToSpeechResponse> SynthesizeToMemoryAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech and yields streaming updates with session lifecycle events.
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="options">Optional synthesis options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>An async enumerable of streaming updates (open → chunk(s) → close).</returns>
    IAsyncEnumerable<TextToSpeechStreamingUpdate> SynthesizeStreamingAsync(
        string text,
        TextToSpeechOptions? options = null,
        CancellationToken cancellationToken = default);
}
