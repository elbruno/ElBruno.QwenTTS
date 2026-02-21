namespace ElBruno.QwenTTS.Audio;

/// <summary>
/// Writes float32 PCM samples to a standard WAV file.
/// Produces 16-bit PCM WAV at the specified sample rate (default 24 kHz).
/// </summary>
public static class WavWriter
{
    /// <summary>
    /// Writes float32 samples to a WAV file as 16-bit PCM.
    /// </summary>
    /// <param name="path">Output file path.</param>
    /// <param name="samples">Float32 PCM samples in [-1.0, 1.0] range.</param>
    /// <param name="sampleRate">Sample rate in Hz (default 24000 for Qwen3-TTS).</param>
    /// <param name="channels">Number of audio channels (default 1 = mono).</param>
    public static void Write(string path, float[] samples, int sampleRate = 24000, int channels = 1)
    {
        const int bitsPerSample = 16;
        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        int dataSize = samples.Length * blockAlign;

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // RIFF header
        writer.Write("RIFF"u8);
        writer.Write(36 + dataSize);
        writer.Write("WAVE"u8);

        // fmt sub-chunk
        writer.Write("fmt "u8);
        writer.Write(16);               // Sub-chunk size
        writer.Write((short)1);          // PCM format
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(byteRate);
        writer.Write((short)blockAlign);
        writer.Write((short)bitsPerSample);

        // data sub-chunk
        writer.Write("data"u8);
        writer.Write(dataSize);

        // Convert float32 → int16 and write
        foreach (var sample in samples)
        {
            var clamped = Math.Clamp(sample, -1.0f, 1.0f);
            var int16 = (short)(clamped * short.MaxValue);
            writer.Write(int16);
        }
    }
}
