using ElBruno.QwenTTS.Audio;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// In-memory PCM synthesis result with metadata and helpers to encode WAV bytes.
/// </summary>
public sealed record TtsAudioResult
{
    /// <summary>Normalized float PCM samples in the [-1.0, 1.0] range.</summary>
    public required ReadOnlyMemory<float> Samples { get; init; }

    /// <summary>Sample rate in Hz.</summary>
    public int SampleRate { get; init; } = WavWriter.DefaultSampleRate;

    /// <summary>Number of audio channels.</summary>
    public int Channels { get; init; } = WavWriter.DefaultChannels;

    /// <summary>PCM bit depth used when encoding WAV output.</summary>
    public int BitsPerSample { get; init; } = WavWriter.DefaultBitsPerSample;

    /// <summary>Timing and output metrics captured for the synthesis request.</summary>
    public TtsSynthesisMetrics Metrics { get; init; } = new();

    /// <summary>Total number of PCM samples across all channels.</summary>
    public int SampleCount => Samples.Length;

    /// <summary>Audio duration derived from the sample count, sample rate, and channels.</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds(SampleCount / (double)(SampleRate * Channels));

    /// <summary>Encodes the PCM samples as a standard 16-bit PCM WAV payload.</summary>
    public ReadOnlyMemory<byte> ToWavBytes() => WavWriter.ToArray(Samples.Span, SampleRate, Channels);
}
