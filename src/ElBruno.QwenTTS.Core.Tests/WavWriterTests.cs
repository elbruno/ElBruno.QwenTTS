using ElBruno.QwenTTS.Audio;

namespace ElBruno.QwenTTS.Core.Tests;

public class WavWriterTests : IDisposable
{
    private readonly string _tempDir;

    public WavWriterTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_wav_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Write_CreatesValidWavFile()
    {
        var path = Path.Combine(_tempDir, "test.wav");
        var samples = new float[24000]; // 1 second of silence at 24kHz
        
        WavWriter.Write(path, samples, sampleRate: 24000);

        Assert.True(File.Exists(path));
        var bytes = File.ReadAllBytes(path);
        
        // WAV header: "RIFF" at offset 0, "WAVE" at offset 8
        Assert.Equal((byte)'R', bytes[0]);
        Assert.Equal((byte)'I', bytes[1]);
        Assert.Equal((byte)'F', bytes[2]);
        Assert.Equal((byte)'F', bytes[3]);
        Assert.Equal((byte)'W', bytes[8]);
        Assert.Equal((byte)'A', bytes[9]);
        Assert.Equal((byte)'V', bytes[10]);
        Assert.Equal((byte)'E', bytes[11]);
    }

    [Fact]
    public void Write_CorrectFileSize()
    {
        var path = Path.Combine(_tempDir, "size_test.wav");
        var samples = new float[4800]; // 0.2 seconds at 24kHz
        
        WavWriter.Write(path, samples, sampleRate: 24000);

        var info = new FileInfo(path);
        // 44 byte header + 4800 samples * 2 bytes (16-bit) = 9644
        Assert.Equal(44 + 4800 * 2, info.Length);
    }

    [Fact]
    public void Write_WithAudioData_NonZeroContent()
    {
        var path = Path.Combine(_tempDir, "audio_test.wav");
        var samples = new float[1000];
        // Generate a simple sine wave
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)Math.Sin(2 * Math.PI * 440 * i / 24000.0) * 0.5f;

        WavWriter.Write(path, samples, sampleRate: 24000);

        var bytes = File.ReadAllBytes(path);
        // Data section should contain non-zero bytes
        var dataSection = bytes.Skip(44).ToArray();
        Assert.Contains(dataSection, b => b != 0);
    }

    [Fact]
    public void Write_SampleRateInHeader()
    {
        var path = Path.Combine(_tempDir, "rate_test.wav");
        var samples = new float[100];
        
        WavWriter.Write(path, samples, sampleRate: 24000);

        var bytes = File.ReadAllBytes(path);
        // Sample rate is at offset 24 (4 bytes, little-endian)
        var sampleRate = BitConverter.ToInt32(bytes, 24);
        Assert.Equal(24000, sampleRate);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
