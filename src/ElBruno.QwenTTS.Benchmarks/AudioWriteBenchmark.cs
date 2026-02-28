using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Benchmarks;

/// <summary>
/// Benchmarks WAV file writing performance by measuring the file I/O portion of TTS synthesis.
/// Uses short, medium, and long texts to generate various audio durations.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporter]
[JsonExporter]
public class AudioWriteBenchmark
{
    private TtsPipeline? _pipeline;
    private string _outputDir = null!;

    [GlobalSetup]
    public void Setup()
    {
        var modelDir = Environment.GetEnvironmentVariable("QWEN_MODEL_DIR");
        if (string.IsNullOrEmpty(modelDir))
        {
            modelDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "ElBruno.QwenTTS", "models");
        }

        if (!Directory.Exists(modelDir))
        {
            throw new InvalidOperationException(
                $"Model directory not found: {modelDir}. " +
                "Run ModelDownloader.DownloadAsync() or set QWEN_MODEL_DIR environment variable.");
        }

        _pipeline = new TtsPipeline(modelDir);
        _outputDir = Path.Combine(Path.GetTempPath(), "qwen_tts_audio_bench");
        Directory.CreateDirectory(_outputDir);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _pipeline?.Dispose();
        if (Directory.Exists(_outputDir))
        {
            try
            {
                Directory.Delete(_outputDir, recursive: true);
            }
            catch { }
        }
    }

    [Benchmark(Description = "Write short audio (~1s)")]
    public async Task WriteShortAudio()
    {
        var outputPath = Path.Combine(_outputDir, "short.wav");
        await _pipeline!.SynthesizeAsync("Hello world.", "Ryan", outputPath, "en");
    }

    [Benchmark(Description = "Write medium audio (~3s)")]
    public async Task WriteMediumAudio()
    {
        var outputPath = Path.Combine(_outputDir, "medium.wav");
        await _pipeline!.SynthesizeAsync(
            "The quick brown fox jumps over the lazy dog.",
            "Ryan", outputPath, "en");
    }

    [Benchmark(Description = "Write long audio (~5s)")]
    public async Task WriteLongAudio()
    {
        var outputPath = Path.Combine(_outputDir, "long.wav");
        await _pipeline!.SynthesizeAsync(
            "The quick brown fox jumps over the lazy dog. " +
            "This sentence is used to benchmark text-to-speech synthesis.",
            "Ryan", outputPath, "en");
    }
}
