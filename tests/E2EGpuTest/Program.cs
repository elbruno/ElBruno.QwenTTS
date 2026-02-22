using System.Diagnostics;
using ElBruno.QwenTTS.Pipeline;

Console.WriteLine("=== ElBruno.QwenTTS End-to-End GPU Test ===");
Console.WriteLine($"Default model dir: {ModelDownloader.DefaultModelDir}");
Console.WriteLine();

var text = "Hello, this is a test of the text to speech pipeline.";
var speaker = "ryan";
var language = "english";
var cpuOutput = "e2e_test_cpu.wav";
var dmlOutput = "e2e_test_dml.wav";
int passed = 0;
int failed = 0;

// --- Test 1: CPU with auto-download ---
Console.WriteLine("=== Test 1: CPU Pipeline (auto-download from HuggingFace) ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var cpuPipeline = await TtsPipeline.CreateAsync(
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");
    Console.WriteLine($"  Available speakers: {string.Join(", ", cpuPipeline.Speakers)}");

    sw.Restart();
    await cpuPipeline.SynthesizeAsync(text, speaker, cpuOutput, language);
    Console.WriteLine($"  Synthesis completed in {sw.Elapsed.TotalSeconds:F1}s");

    var fi = new FileInfo(cpuOutput);
    if (fi.Exists && fi.Length > 1000)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        Console.WriteLine("  ✓ CPU: PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  ✗ CPU: FAIL - output file too small or missing");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ CPU: FAIL - {ex.GetType().Name}: {ex.Message}");
    failed++;
}

Console.WriteLine();

// --- Test 2: DirectML hybrid (GPU LM + CPU vocoder) with auto-download ---
Console.WriteLine("=== Test 2: DirectML Hybrid Pipeline (GPU LM + CPU vocoder) ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var dmlPipeline = await TtsPipeline.CreateAsync(
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
        vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");

    sw.Restart();
    await dmlPipeline.SynthesizeAsync(text, speaker, dmlOutput, language);
    Console.WriteLine($"  Synthesis completed in {sw.Elapsed.TotalSeconds:F1}s");

    var fi = new FileInfo(dmlOutput);
    if (fi.Exists && fi.Length > 1000)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        Console.WriteLine("  ✓ DirectML: PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  ✗ DirectML: FAIL - output file too small or missing");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ DirectML: FAIL - {ex.GetType().Name}: {ex.Message}");
    failed++;
}

Console.WriteLine();
Console.WriteLine($"=== Results: {passed} passed, {failed} failed ===");

// Cleanup test files
foreach (var f in new[] { cpuOutput, dmlOutput })
    if (File.Exists(f)) File.Delete(f);

return failed > 0 ? 1 : 0;
