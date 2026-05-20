using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS;

/// <summary>
/// CLI entry point for Qwen3-TTS inference.
/// Usage: QwenTTS --model-dir ./models --text "Hello" --speaker Ryan --output hello.wav [--language english] [--instruct "speak happily"] [--variant 1.7b]
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
        var variantStr = GetArg(args, "--variant");

        if (string.IsNullOrEmpty(modelDir))
        {
            Console.Error.WriteLine("Error: --model-dir is required");
            PrintUsage();
            return 1;
        }

        if (string.IsNullOrEmpty(text))
        {
            Console.Error.WriteLine("Error: --text is required");
            PrintUsage();
            return 1;
        }

        var variant = ParseVariant(variantStr);

        Console.WriteLine($"Model:    {modelDir}");
        Console.WriteLine($"Variant:  {variant}");
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
                progress: new Progress<string>(msg => Console.WriteLine($"  {msg}")),
                variant: variant);
            await pipeline.SynthesizeAsync(text, speaker, output, language, instruct);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static QwenModelVariant ParseVariant(string? value)
    {
        if (string.IsNullOrEmpty(value))
            return QwenModelVariant.Qwen06B;

        return value.ToLowerInvariant() switch
        {
            "0.6b" or "06b" or "0.6" => QwenModelVariant.Qwen06B,
            "1.7b" or "17b" or "1.7" => QwenModelVariant.Qwen17B,
            _ => throw new ArgumentException($"Unknown variant '{value}'. Valid values: 0.6b, 1.7b")
        };
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("Usage: QwenTTS --model-dir ./models --text \"Hello\" --speaker Ryan --output hello.wav [--language english] [--instruct \"speak happily\"] [--variant 1.7b]");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Options:");
        Console.Error.WriteLine("  --model-dir   Path to model directory (required)");
        Console.Error.WriteLine("  --text        Text to synthesize (required)");
        Console.Error.WriteLine("  --speaker     Speaker name (default: Ryan)");
        Console.Error.WriteLine("  --output      Output WAV file (default: output.wav)");
        Console.Error.WriteLine("  --language    Language: auto, english, spanish, chinese, japanese, korean, russian, etc. (default: auto)");
        Console.Error.WriteLine("  --instruct    Style instruction, e.g. \"Read with warmth\" (1.7B only)");
        Console.Error.WriteLine("  --variant     Model variant: 0.6b (default), 1.7b");
    }

    private static string? GetArg(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }
}
