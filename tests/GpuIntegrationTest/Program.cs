using System.Diagnostics;
using ElBruno.QwenTTS.Pipeline;
using Microsoft.ML.OnnxRuntime;

var modelDir = args.Length > 0 ? args[0] : @"C:\src\qwen-labs-cs\python\onnx_runtime_dml";
var text = "Hello, this is a GPU test.";

Console.WriteLine($"Model dir: {modelDir}");
Console.WriteLine($"Text: {text}");
Console.WriteLine();

// --- CPU Test ---
Console.WriteLine("=== CPU Test ===");
try
{
    var sw = Stopwatch.StartNew();
    using var cpuPipeline = new TtsPipeline(modelDir, OrtSessionHelper.CreateCpuOptions);
    Console.WriteLine($"CPU pipeline created in {sw.ElapsedMilliseconds} ms");

    sw.Restart();
    await cpuPipeline.SynthesizeAsync(text, "ryan", "test_cpu.wav", "english");
    Console.WriteLine($"CPU synthesis completed in {sw.ElapsedMilliseconds} ms");

    var cpuFile = new FileInfo("test_cpu.wav");
    Console.WriteLine($"CPU output: {cpuFile.Name} ({cpuFile.Length} bytes)");
    Console.WriteLine("CPU: PASS");
}
catch (Exception ex)
{
    Console.WriteLine($"CPU: FAIL - {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine();

// --- DirectML Test (hybrid: DML for LM, CPU for vocoder) ---
Console.WriteLine("=== DirectML Test (hybrid: GPU LM + CPU vocoder) ===");
try
{
    var sw = Stopwatch.StartNew();
    using var dmlPipeline = new TtsPipeline(
        modelDir,
        sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
        vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);
    Console.WriteLine($"DirectML pipeline created in {sw.ElapsedMilliseconds} ms");

    sw.Restart();
    await dmlPipeline.SynthesizeAsync(text, "ryan", "test_dml.wav", "english");
    Console.WriteLine($"DirectML synthesis completed in {sw.ElapsedMilliseconds} ms");

    var dmlFile = new FileInfo("test_dml.wav");
    Console.WriteLine($"DirectML output: {dmlFile.Name} ({dmlFile.Length} bytes)");
    Console.WriteLine("DirectML: PASS");
}
catch (Exception ex)
{
    Console.WriteLine($"DirectML: FAIL - {ex.GetType().Name}: {ex.Message}");
}

// Cleanup
foreach (var f in new[] { "test_cpu.wav", "test_dml.wav" })
    if (File.Exists(f)) File.Delete(f);

Console.WriteLine("\nDone.");

