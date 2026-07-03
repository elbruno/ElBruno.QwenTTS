namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Timing and output metadata captured for a single synthesis request.
/// </summary>
public sealed class TtsSynthesisMetrics
{
    /// <summary>Time spent waiting for an available synthesis slot.</summary>
    public TimeSpan QueueLatency { get; init; }

    /// <summary>Time from request start until waveform samples are ready in memory.</summary>
    public TimeSpan FirstAudioLatency { get; init; }

    /// <summary>Total time from request start until synthesis metrics are captured after vocoder decode.</summary>
    public TimeSpan TotalLatency { get; init; }

    /// <summary>Number of generated codec frames.</summary>
    public int GeneratedFrames { get; init; }

    /// <summary>Number of PCM samples written to the output waveform.</summary>
    public int OutputSamples { get; init; }
}
