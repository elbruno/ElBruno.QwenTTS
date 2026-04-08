using System.Diagnostics;
using ElBruno.QwenTTS.Pipeline;

var promptMode = args.Contains("--prompt", StringComparer.OrdinalIgnoreCase);

static bool PromptYesNo(string message, bool promptMode)
{
    if (!promptMode)
        return true; // auto-run when --prompt is not specified

    Console.Write($"{message} [y/n] (default: y): ");
    var input = Console.ReadLine()?.Trim() ?? "";
    return input.Length == 0
        || input.Equals("y", StringComparison.OrdinalIgnoreCase)
        || input.Equals("yes", StringComparison.OrdinalIgnoreCase);
}

Console.WriteLine("=== Issue #30: 1.7B Model Noise Diagnostic Test ===");
Console.WriteLine($"Default model dir: {ModelDownloader.DefaultModelDir}");
Console.WriteLine();
Console.WriteLine("This test compares 0.6B (working baseline) vs 1.7B (reported noise issue)");
Console.WriteLine("Testing both English and Chinese text, with speaker 'vivian' for Chinese.");
Console.WriteLine();

var englishText = "Hello, this is a test of the text to speech pipeline.";
var chineseText = "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"; // From issue #30
var speaker = "vivian";
var englishLanguage = "english";
var chineseLanguage = "chinese";

var outputs = new List<(string model, string lang, string path, long size, double duration)>();

// Prompt for 0.6B tests
if (PromptYesNo("Run 0.6B baseline tests?", promptMode))
{
// Test 1: 0.6B English (baseline)
Console.WriteLine("=== Test 1: 0.6B Model - English ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var pipeline06b = await TtsPipeline.CreateAsync(
        variant: QwenModelVariant.Qwen06B,
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");
    
    var outputPath = "issue30_06b_english.wav";
    sw.Restart();
    await pipeline06b.SynthesizeAsync(englishText, speaker, outputPath, englishLanguage);
    var synthTime = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"  Synthesis completed in {synthTime:F1}s");

    var fi = new FileInfo(outputPath);
    if (fi.Exists)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        outputs.Add(("0.6B", "English", outputPath, fi.Length, synthTime));
        Console.WriteLine("  ✓ PASS");
    }
    else
    {
        Console.WriteLine("  ✗ FAIL - output file missing");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ FAIL - {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine();

// Test 2: 0.6B Chinese with vivian (baseline)
Console.WriteLine("=== Test 2: 0.6B Model - Chinese (speaker: vivian) ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var pipeline06b = await TtsPipeline.CreateAsync(
        variant: QwenModelVariant.Qwen06B,
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");
    
    var outputPath = "issue30_06b_chinese.wav";
    sw.Restart();
    await pipeline06b.SynthesizeAsync(chineseText, speaker, outputPath, chineseLanguage);
    var synthTime = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"  Synthesis completed in {synthTime:F1}s");

    var fi = new FileInfo(outputPath);
    if (fi.Exists)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        outputs.Add(("0.6B", "Chinese", outputPath, fi.Length, synthTime));
        Console.WriteLine("  ✓ PASS");
    }
    else
    {
        Console.WriteLine("  ✗ FAIL - output file missing");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ FAIL - {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine();
}
else
{
    Console.WriteLine("Skipping 0.6B tests.");
    Console.WriteLine();
}

// Prompt for 1.7B tests
if (PromptYesNo("Run 1.7B tests?", promptMode))
{
// Test 3: 1.7B English (test case)
Console.WriteLine("=== Test 3: 1.7B Model - English ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var pipeline17b = await TtsPipeline.CreateAsync(
        variant: QwenModelVariant.Qwen17B,
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");
    
    var outputPath = "issue30_17b_english.wav";
    sw.Restart();
    await pipeline17b.SynthesizeAsync(englishText, speaker, outputPath, englishLanguage);
    var synthTime = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"  Synthesis completed in {synthTime:F1}s");

    var fi = new FileInfo(outputPath);
    if (fi.Exists)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        outputs.Add(("1.7B", "English", outputPath, fi.Length, synthTime));
        Console.WriteLine("  ✓ PASS");
    }
    else
    {
        Console.WriteLine("  ✗ FAIL - output file missing");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ FAIL - {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine();

// Test 4: 1.7B Chinese with vivian (issue #30 reproduction)
Console.WriteLine("=== Test 4: 1.7B Model - Chinese (speaker: vivian) [Issue #30] ===");
try
{
    var sw = Stopwatch.StartNew();
    var progress = new Progress<string>(msg => Console.WriteLine($"  [Progress] {msg}"));

    using var pipeline17b = await TtsPipeline.CreateAsync(
        variant: QwenModelVariant.Qwen17B,
        progress: progress,
        sessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

    Console.WriteLine($"  Pipeline created in {sw.Elapsed.TotalSeconds:F1}s");
    
    var outputPath = "issue30_17b_chinese.wav";
    sw.Restart();
    await pipeline17b.SynthesizeAsync(chineseText, speaker, outputPath, chineseLanguage);
    var synthTime = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"  Synthesis completed in {synthTime:F1}s");

    var fi = new FileInfo(outputPath);
    if (fi.Exists)
    {
        Console.WriteLine($"  Output: {fi.Name} ({fi.Length:N0} bytes)");
        outputs.Add(("1.7B", "Chinese", outputPath, fi.Length, synthTime));
        Console.WriteLine("  ✓ PASS");
    }
    else
    {
        Console.WriteLine("  ✗ FAIL - output file missing");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  ✗ FAIL - {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine();
}
else
{
    Console.WriteLine("Skipping 1.7B tests.");
    Console.WriteLine();
}

Console.WriteLine("=== Comparison Summary ===");
Console.WriteLine();
Console.WriteLine($"{"Model",-8} {"Language",-10} {"File",-30} {"Size",-15} {"Duration"}");
Console.WriteLine(new string('-', 80));
foreach (var (model, lang, path, size, duration) in outputs)
{
    Console.WriteLine($"{model,-8} {lang,-10} {Path.GetFileName(path),-30} {size,10:N0} bytes {duration,6:F1}s");
}

Console.WriteLine();
Console.WriteLine("=== Manual Validation Required ===");
Console.WriteLine("Please listen to the generated WAV files:");
foreach (var (model, lang, path, _, _) in outputs)
{
    Console.WriteLine($"  - {path}  ({model} - {lang})");
}
Console.WriteLine();
Console.WriteLine("Expected: 0.6B files should sound clear.");
Console.WriteLine("Issue: 1.7B files may sound like noise (as reported in #30).");
Console.WriteLine();

return 0;
