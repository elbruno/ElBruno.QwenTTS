using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS;

/// <summary>
/// CLI entry point for Qwen3-TTS inference.
/// Usage: QwenTTS --model-dir ./models --text "Hello" --speaker Ryan --output hello.wav [--language english] [--instruct "speak happily"]
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        var modelDir = GetArg(args, "--model-dir");
        var text = GetArg(args, "--text");
        var speaker = GetArg(args, "--speaker") ?? "Ryan";
        var output = GetArg(args, "--output") ?? "output.wav";
        var language = GetArg(args, "--language") ?? "auto";
        var instruct = GetArg(args, "--instruct");

        if (string.IsNullOrEmpty(modelDir))
        {
            Console.Error.WriteLine("Error: --model-dir is required");
            Console.Error.WriteLine("Usage: QwenTTS --model-dir ./models --text \"Hello\" --speaker Ryan --output hello.wav [--language english] [--instruct \"speak happily\"]");
            return 1;
        }

        if (string.IsNullOrEmpty(text))
        {
            Console.Error.WriteLine("Error: --text is required");
            Console.Error.WriteLine("Usage: QwenTTS --model-dir ./models --text \"Hello\" --speaker Ryan --output hello.wav [--language english] [--instruct \"speak happily\"]");
            return 1;
        }

        Console.WriteLine($"Model:    {modelDir}");
        Console.WriteLine($"Text:     {text}");
        Console.WriteLine($"Speaker:  {speaker}");
        Console.WriteLine($"Language: {language}");
        Console.WriteLine($"Output:   {output}");
        if (instruct is not null)
            Console.WriteLine($"Instruct: {instruct}");

        try
        {
            // Auto-download models if not present
            using var pipeline = await TtsPipeline.CreateAsync(modelDir,
                progress: new Progress<string>(msg => Console.WriteLine($"  {msg}")));
            await pipeline.SynthesizeAsync(text, speaker, output, language, instruct);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static string? GetArg(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }
}
