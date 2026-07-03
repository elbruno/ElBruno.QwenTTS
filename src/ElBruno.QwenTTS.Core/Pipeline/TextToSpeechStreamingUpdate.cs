namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Kind of streaming update during text-to-speech synthesis.
/// </summary>
public enum TextToSpeechUpdateKind
{
    /// <summary>Synthesis session has opened.</summary>
    SessionOpen,

    /// <summary>An audio chunk is available.</summary>
    AudioChunk,

    /// <summary>Synthesis session has closed.</summary>
    SessionClose
}

/// <summary>
/// A streaming update from a text-to-speech synthesis operation.
/// </summary>
public sealed class TextToSpeechStreamingUpdate
{
    /// <summary>The kind of update.</summary>
    public required TextToSpeechUpdateKind Kind { get; init; }

    /// <summary>Audio data for <see cref="TextToSpeechUpdateKind.AudioChunk"/> updates. Null for other kinds.</summary>
    public byte[]? AudioData { get; init; }

    /// <summary>Sample rate in Hz for the stream. Present on session open and audio chunk updates.</summary>
    public int? SampleRate { get; init; }

    /// <summary>MIME type for the emitted audio stream (for example, <c>audio/wav</c>).</summary>
    public string? MediaType { get; init; }

    /// <summary>Number of audio channels in the stream.</summary>
    public int? Channels { get; init; }

    /// <summary>Bits per sample for PCM audio.</summary>
    public int? BitsPerSample { get; init; }

    /// <summary>
    /// True when audio is emitted progressively during inference; false when emitted as ordered chunks after generation completes.
    /// </summary>
    public bool IsProgressive { get; init; }

    /// <summary>Per-request synthesis metrics, typically attached to the final session close update.</summary>
    public TtsSynthesisMetrics? Metrics { get; init; }
}
