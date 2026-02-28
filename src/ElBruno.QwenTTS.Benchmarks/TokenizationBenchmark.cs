using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Benchmarks;

[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
[MarkdownExporter]
[JsonExporter]
public class TokenizationBenchmark
{
    private TtsPipeline? _pipeline;
    private string _englishShort = null!;
    private string _englishLong = null!;
    private string _cjkText = null!;
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

        _outputDir = Path.Combine(Path.GetTempPath(), "qwen_tts_tokenization_bench");
        Directory.CreateDirectory(_outputDir);

        // Prepare test inputs
        _englishShort = "Hello, this is a short text for tokenization benchmarking. It contains about one hundred characters!";
        _englishLong = string.Join(" ", Enumerable.Repeat(_englishShort, 10)); // ~1000 chars
        _cjkText = "你好，这是一段中文文本用于基准测试。文本包含大约一百个字符。北京天气很好，今天是个晴天。机器学习和深度学习技术正在快速发展。人工智能应用已经深入各个领域。文本到语音合成是重要技术之一。";
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

    [Benchmark(Description = "Process English 100 chars")]
    public async Task ProcessEnglishShort()
    {
        var outputPath = Path.Combine(_outputDir, "short.wav");
        await _pipeline!.SynthesizeAsync(_englishShort, "Ryan", outputPath, "en");
    }

    [Benchmark(Description = "Process English 1000 chars")]
    public async Task ProcessEnglishLong()
    {
        var outputPath = Path.Combine(_outputDir, "long.wav");
        await _pipeline!.SynthesizeAsync(_englishLong, "Ryan", outputPath, "en");
    }

    [Benchmark(Description = "Process CJK 100+ chars")]
    public async Task ProcessCjk()
    {
        var outputPath = Path.Combine(_outputDir, "cjk.wav");
        await _pipeline!.SynthesizeAsync(_cjkText, "Eric", outputPath, "zh");
    }
}
