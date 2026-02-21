using QwenTTS.Pipeline;

namespace QwenTTS;

/// <summary>
/// CLI entry point for Qwen3-TTS inference.
/// Usage: QwenTTS --text "Hello" --speaker Ryan --output hello.wav [--instruct "speak happily"]
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        var text = GetArg(args, "--text");
        var speaker = GetArg(args, "--speaker") ?? "Ryan";
        var output = GetArg(args, "--output") ?? "output.wav";
        var instruct = GetArg(args, "--instruct");

        if (string.IsNullOrEmpty(text))
        {
            Console.Error.WriteLine("Usage: QwenTTS --text \"Hello\" --speaker Ryan --output hello.wav [--instruct \"speak happily\"]");
            return 1;
        }

        Console.WriteLine($"Text:    {text}");
        Console.WriteLine($"Speaker: {speaker}");
        Console.WriteLine($"Output:  {output}");
        if (instruct is not null)
            Console.WriteLine($"Instruct: {instruct}");

        // TODO: Initialize the TTS pipeline with model paths and run inference
        // var pipeline = new TtsPipeline(modelDir);
        // await pipeline.SynthesizeAsync(text, speaker, output, instruct);

        Console.WriteLine("Pipeline not yet wired — ONNX models required.");
        return 0;
    }

    private static string? GetArg(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }
}
