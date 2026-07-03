using System.Buffers.Binary;

namespace ElBruno.QwenTTS.Audio;

/// <summary>
/// Writes float32 PCM samples to a standard WAV file.
/// Produces 16-bit PCM WAV at the specified sample rate (default 24 kHz).
/// </summary>
internal static class WavWriter
{
    internal const int DefaultSampleRate = 24000;
    internal const int DefaultChannels = 1;
    internal const int DefaultBitsPerSample = 16;
    internal const int HeaderSize = 44;

    /// <summary>
    /// Writes float32 samples to a WAV file as 16-bit PCM.
    /// </summary>
    /// <param name="path">Output file path.</param>
    /// <param name="samples">Float32 PCM samples in [-1.0, 1.0] range.</param>
    /// <param name="sampleRate">Sample rate in Hz (default 24000 for Qwen3-TTS).</param>
    /// <param name="channels">Number of audio channels (default 1 = mono).</param>
    public static void Write(string path, float[] samples, int sampleRate = DefaultSampleRate, int channels = DefaultChannels)
    {
        using var stream = File.Create(path);
        stream.Write(CreateHeader(samples.Length, sampleRate, channels));
        WritePcmSamples(stream, samples);
    }

    /// <summary>
    /// Enumerates ordered WAV chunks where simple byte concatenation reconstructs a valid WAV file.
    /// The first chunk contains the WAV header plus initial PCM bytes; later chunks contain PCM bytes only.
    /// </summary>
    public static IEnumerable<byte[]> EnumerateWavChunks(
        float[] samples,
        int sampleRate = DefaultSampleRate,
        int channels = DefaultChannels,
        int maxChunkBytes = 32 * 1024)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (maxChunkBytes <= HeaderSize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxChunkBytes),
                $"Chunk size must be greater than the {HeaderSize}-byte WAV header.");
        }

        var totalPcmBytes = GetPcmByteLength(samples.Length);
        var header = CreateHeader(samples.Length, sampleRate, channels);

        var firstChunkPcmBytes = Math.Min(totalPcmBytes, maxChunkBytes - header.Length);
        var firstChunk = new byte[header.Length + firstChunkPcmBytes];
        header.CopyTo(firstChunk, 0);
        WritePcmSamples(samples, 0, firstChunkPcmBytes / sizeof(short), firstChunk.AsSpan(header.Length));
        yield return firstChunk;

        var samplesPerChunk = Math.Max(1, maxChunkBytes / sizeof(short));
        var writtenSamples = firstChunkPcmBytes / sizeof(short);
        while (writtenSamples < samples.Length)
        {
            var chunkSampleCount = Math.Min(samplesPerChunk, samples.Length - writtenSamples);
            var chunk = new byte[GetPcmByteLength(chunkSampleCount)];
            WritePcmSamples(samples, writtenSamples, chunkSampleCount, chunk);
            yield return chunk;
            writtenSamples += chunkSampleCount;
        }
    }

    private static byte[] CreateHeader(int sampleCount, int sampleRate, int channels)
    {
        var dataSize = GetPcmByteLength(sampleCount);
        var header = new byte[HeaderSize];
        var riffSize = 36 + dataSize;
        var byteRate = sampleRate * channels * DefaultBitsPerSample / 8;
        var blockAlign = channels * DefaultBitsPerSample / 8;

        "RIFF"u8.CopyTo(header.AsSpan(0, 4));
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(4, 4), riffSize);
        "WAVE"u8.CopyTo(header.AsSpan(8, 4));
        "fmt "u8.CopyTo(header.AsSpan(12, 4));
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(16, 4), 16);
        BinaryPrimitives.WriteInt16LittleEndian(header.AsSpan(20, 2), 1);
        BinaryPrimitives.WriteInt16LittleEndian(header.AsSpan(22, 2), checked((short)channels));
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(24, 4), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(28, 4), byteRate);
        BinaryPrimitives.WriteInt16LittleEndian(header.AsSpan(32, 2), checked((short)blockAlign));
        BinaryPrimitives.WriteInt16LittleEndian(header.AsSpan(34, 2), DefaultBitsPerSample);
        "data"u8.CopyTo(header.AsSpan(36, 4));
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(40, 4), dataSize);

        return header;
    }

    private static int GetPcmByteLength(int sampleCount) => checked(sampleCount * sizeof(short));

    private static void WritePcmSamples(Stream stream, float[] samples)
    {
        var buffer = new byte[Math.Min(samples.Length, 4096) * sizeof(short)];
        var offset = 0;
        while (offset < samples.Length)
        {
            var sampleCount = Math.Min(buffer.Length / sizeof(short), samples.Length - offset);
            var span = buffer.AsSpan(0, GetPcmByteLength(sampleCount));
            WritePcmSamples(samples, offset, sampleCount, span);
            stream.Write(span);
            offset += sampleCount;
        }
    }

    private static void WritePcmSamples(float[] samples, int startSample, int sampleCount, Span<byte> destination)
    {
        for (var index = 0; index < sampleCount; index++)
        {
            var clamped = Math.Clamp(samples[startSample + index], -1.0f, 1.0f);
            var int16 = (short)(clamped * short.MaxValue);
            BinaryPrimitives.WriteInt16LittleEndian(destination.Slice(index * sizeof(short), sizeof(short)), int16);
        }
    }
}
