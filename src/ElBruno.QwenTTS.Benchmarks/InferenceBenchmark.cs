using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporter]
[JsonExporter]
public class InferenceBenchmark
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

        // Pre-load models (warm cache)
        _pipeline = new TtsPipeline(modelDir);

        // Setup output directory for WAV files
        _outputDir = Path.Combine(Path.GetTempPath(), "qwen_tts_benchmarks");
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
            catch
            {
                // Best effort cleanup
            }
        }
    }

    [Benchmark(Description = "TTS Short (10 words)")]
    public async Task SynthesizeShortText()
    {
        var outputPath = Path.Combine(_outputDir, "short.wav");
        await _pipeline!.SynthesizeAsync(
            "Hello world, this is a short test sentence.",
            "Ryan",
            outputPath,
            "en");
    }

    [Benchmark(Description = "TTS Medium (30 words)")]
    public async Task SynthesizeMediumText()
    {
        var outputPath = Path.Combine(_outputDir, "medium.wav");
        await _pipeline!.SynthesizeAsync(
            "The quick brown fox jumps over the lazy dog. " +
            "This sentence is used to benchmark text-to-speech synthesis performance. " +
            "It contains approximately thirty words in total.",
            "Ryan",
            outputPath,
            "en");
    }

    [Benchmark(Description = "TTS CJK (short)")]
    public async Task SynthesizeCjkText()
    {
        var outputPath = Path.Combine(_outputDir, "cjk.wav");
        await _pipeline!.SynthesizeAsync(
            "你好，这是一段中文文本用于测试语音合成性能。",
            "Eric",
            outputPath,
            "zh");
    }
}
