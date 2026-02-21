using System.Text;
using System.Text.RegularExpressions;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.FileReader;

/// <summary>
/// CLI app that reads a text or SRT file and generates speech audio for each segment.
/// Usage: ElBruno.QwenTTS.FileReader --model-dir ./models --input file.txt --speaker ryan [--language auto] [--output-dir ./output] [--instruct "speak calmly"]
/// </summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        var modelDir = GetArg(args, "--model-dir");
        var inputFile = GetArg(args, "--input");
        var speaker = GetArg(args, "--speaker") ?? "Ryan";
        var outputDir = GetArg(args, "--output-dir") ?? "output";
        var language = GetArg(args, "--language") ?? "auto";
        var instruct = GetArg(args, "--instruct");

        if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(inputFile))
        {
            Console.Error.WriteLine("Usage: ElBruno.QwenTTS.FileReader --model-dir ./models --input file.txt --speaker ryan [--language auto] [--output-dir ./output] [--instruct \"speak calmly\"]");
            return 1;
        }

        if (!File.Exists(inputFile))
        {
            Console.Error.WriteLine($"Error: file not found: {inputFile}");
            return 1;
        }

        Directory.CreateDirectory(outputDir);

        var ext = Path.GetExtension(inputFile).ToLowerInvariant();
        var segments = ext switch
        {
            ".srt" => ParseSrt(inputFile),
            _ => ParseText(inputFile),
        };

        if (segments.Count == 0)
        {
            Console.Error.WriteLine("Error: no text segments found in input file");
            return 1;
        }

        Console.WriteLine($"Model:    {modelDir}");
        Console.WriteLine($"Input:    {inputFile} ({ext})");
        Console.WriteLine($"Speaker:  {speaker}");
        Console.WriteLine($"Language: {language}");
        Console.WriteLine($"Output:   {outputDir}");
        Console.WriteLine($"Segments: {segments.Count}");
        Console.WriteLine();

        try
        {
            using var pipeline = new TtsPipeline(modelDir);
            var baseName = Path.GetFileNameWithoutExtension(inputFile);

            for (int i = 0; i < segments.Count; i++)
            {
                var (label, text) = segments[i];
                var outFile = Path.Combine(outputDir, $"{baseName}_{i + 1:D3}.wav");
                var displayText = text.Length > 80 ? text[..77] + "..." : text;

                Console.WriteLine($"[{i + 1}/{segments.Count}] {label}");
                Console.WriteLine($"  Text: {displayText}");

                await pipeline.SynthesizeAsync(text, speaker, outFile, language, instruct);
                Console.WriteLine($"  Saved: {outFile}");
                Console.WriteLine();
            }

            Console.WriteLine($"Done! Generated {segments.Count} audio files in {outputDir}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    /// <summary>Parse a plain text file into segments split by blank lines (paragraphs).</summary>
    static List<(string Label, string Text)> ParseText(string path)
    {
        var content = File.ReadAllText(path, Encoding.UTF8);
        var paragraphs = Regex.Split(content, @"\r?\n\s*\r?\n")
            .Select(p => p.Trim())
            .Where(p => p.Length > 0)
            .ToList();

        return paragraphs
            .Select((p, i) => ($"Paragraph {i + 1}", StripMarkdown(p)))
            .ToList();
    }

    /// <summary>Parse an SRT subtitle file into segments.</summary>
    static List<(string Label, string Text)> ParseSrt(string path)
    {
        var content = File.ReadAllText(path, Encoding.UTF8);
        var entries = Regex.Split(content, @"\r?\n\r?\n")
            .Where(e => e.Trim().Length > 0)
            .ToList();

        var segments = new List<(string Label, string Text)>();

        foreach (var entry in entries)
        {
            var lines = entry.Trim().Split('\n').Select(l => l.Trim()).ToArray();
            if (lines.Length < 3) continue;

            // Line 0: sequence number, Line 1: timecodes, Lines 2+: text
            var text = string.Join(" ", lines.Skip(2));
            text = Regex.Replace(text, @"<[^>]+>", "").Trim(); // strip HTML tags
            if (text.Length > 0)
                segments.Add(($"SRT #{lines[0]}", text));
        }

        return segments;
    }

    /// <summary>Strip common markdown formatting for cleaner speech synthesis.</summary>
    static string StripMarkdown(string text)
    {
        text = Regex.Replace(text, @"^#+\s*", "", RegexOptions.Multiline); // headings
        text = Regex.Replace(text, @"\*\*([^*]+)\*\*", "$1");             // bold
        text = Regex.Replace(text, @"\*([^*]+)\*", "$1");                 // italic
        text = Regex.Replace(text, @"`([^`]+)`", "$1");                   // inline code
        text = Regex.Replace(text, @"\[([^\]]+)\]\([^)]+\)", "$1");       // links
        text = Regex.Replace(text, @"^[-*]\s+", "", RegexOptions.Multiline); // list items
        text = Regex.Replace(text, @"^>\s*", "", RegexOptions.Multiline); // blockquotes
        text = Regex.Replace(text, @"---+", "", RegexOptions.Multiline);  // horizontal rules
        text = Regex.Replace(text, @"\s+", " ").Trim();
        return text;
    }

    static string? GetArg(string[] args, string name)
    {
        var index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }
}
