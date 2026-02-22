using Xunit;
using ElBruno.QwenTTS.VoiceCloning.Audio;
using ElBruno.QwenTTS.VoiceCloning.Models;
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

namespace ElBruno.QwenTTS.VoiceCloning.Tests;

public class MelSpectrogramTests : IDisposable
{
    private readonly string _tempDir;

    public MelSpectrogramTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"vc_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Extract_ReturnsCorrectShape()
    {
        // 1 second of silence at 24 kHz
        int sampleRate = 24000;
        int nSamples = sampleRate;
        var samples = new float[nSamples];

        var mel = MelSpectrogram.Extract(samples, sampleRate);

        // T_mel = 1 + (24000 - 1024) / 256 = 90 frames (approx)
        Assert.True(mel.GetLength(0) > 0, "Should have at least one frame");
        Assert.Equal(128, mel.GetLength(1));  // 128 mel bins
    }

    [Fact]
    public void Extract_WithSineWave_HasNonZeroEnergy()
    {
        int sampleRate = 24000;
        int nSamples = sampleRate; // 1 second
        var samples = new float[nSamples];

        // Generate 440 Hz sine wave
        for (int i = 0; i < nSamples; i++)
            samples[i] = MathF.Sin(2f * MathF.PI * 440f * i / sampleRate) * 0.5f;

        var mel = MelSpectrogram.Extract(samples, sampleRate);

        // At least some mel bins should have non-trivial energy
        float maxVal = float.MinValue;
        for (int t = 0; t < mel.GetLength(0); t++)
            for (int m = 0; m < mel.GetLength(1); m++)
                maxVal = Math.Max(maxVal, mel[t, m]);

        Assert.True(maxVal > -20f, "Sine wave should produce significant mel energy");
    }

    [Fact]
    public void Extract_ShortAudio_StillWorks()
    {
        // Very short audio: 256 samples (one hop)
        var samples = new float[256];
        for (int i = 0; i < 256; i++)
            samples[i] = MathF.Sin(2f * MathF.PI * 1000f * i / 24000);

        var mel = MelSpectrogram.Extract(samples);

        Assert.True(mel.GetLength(0) >= 1, "Should produce at least one frame");
        Assert.Equal(128, mel.GetLength(1));
    }

    [Fact]
    public void FromWavFile_ReadsAndExtracts()
    {
        // Create a simple WAV file
        var wavPath = Path.Combine(_tempDir, "test.wav");
        int sampleRate = 24000;
        int nSamples = sampleRate; // 1 second
        var samples = new float[nSamples];
        for (int i = 0; i < nSamples; i++)
            samples[i] = MathF.Sin(2f * MathF.PI * 440f * i / sampleRate) * 0.3f;

        ElBruno.QwenTTS.Audio.WavWriter.Write(wavPath, samples, sampleRate);

        var mel = MelSpectrogram.FromWavFile(wavPath);

        Assert.True(mel.GetLength(0) > 0);
        Assert.Equal(128, mel.GetLength(1));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}

public class VoiceCloningDownloaderTests : IDisposable
{
    private readonly string _tempDir;

    public VoiceCloningDownloaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"vc_dl_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void IsModelDownloaded_EmptyDir_ReturnsFalse()
    {
        Assert.False(VoiceCloningDownloader.IsModelDownloaded(_tempDir));
    }

    [Fact]
    public void DefaultModelDir_IsUnderElBruno()
    {
        var dir = VoiceCloningDownloader.DefaultModelDir;
        Assert.Contains("ElBruno", dir);
        Assert.Contains("QwenTTS-Base", dir);
    }

    [Fact]
    public async Task DownloadModelAsync_CancellationPreventsDownload()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            VoiceCloningDownloader.DownloadModelAsync(_tempDir, cancellationToken: cts.Token));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}

public class SpeakerEncoderTests
{
    [Fact]
    public void Constructor_AcceptsModelPath()
    {
        // Just verify the constructor doesn't throw (lazy loading)
        var encoder = new SpeakerEncoder("nonexistent.onnx");
        Assert.Equal(1024, encoder.EmbeddingDim);
        Assert.Equal(128, encoder.MelDim);
        encoder.Dispose();
    }
}

public class VoiceClonePipelineTests
{
    [Fact]
    public void Constructor_ThrowsIfSpeakerEncoderMissing()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"vc_pipe_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            Assert.Throws<FileNotFoundException>(() => new VoiceClonePipeline(tempDir));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
