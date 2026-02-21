using QwenTTS.Audio;
using QwenTTS.Models;

namespace QwenTTS.Pipeline;

/// <summary>
/// Orchestrates the full TTS pipeline: text → tokenize → LM → vocoder → WAV.
/// </summary>
public sealed class TtsPipeline : IDisposable
{
    private readonly TextTokenizer _tokenizer;
    private readonly LanguageModel _languageModel;
    private readonly Vocoder _vocoder;
    private readonly EmbeddingStore _embeddings;

    public TtsPipeline(string modelDir)
    {
        var tokenizerDir = Path.Combine(modelDir, "tokenizer");
        var embeddingsDir = Path.Combine(modelDir, "embeddings");
        var configPath = Path.Combine(embeddingsDir, "config.json");

        _tokenizer = new TextTokenizer(tokenizerDir);
        _embeddings = new EmbeddingStore(embeddingsDir, configPath);
        _languageModel = new LanguageModel(modelDir, _embeddings);
        _vocoder = new Vocoder(Path.Combine(modelDir, "vocoder.onnx"));
    }

    /// <summary>Available speaker names from the model.</summary>
    public IReadOnlyCollection<string> Speakers => _embeddings.GetAvailableSpeakers();

    public async Task SynthesizeAsync(string text, string speaker, string outputPath, 
                                     string language = "auto", string? instruct = null)
    {
        // Build prompt using tokenizer
        var tokenIds = _tokenizer.BuildCustomVoicePrompt(text, speaker, language, instruct);

        Console.WriteLine($"Generating speech ({tokenIds.Length} input tokens)...");

        // Generate audio codes via LM
        var codes = _languageModel.Generate(tokenIds, speaker, language);
        
        int timesteps = codes.GetLength(2);
        Console.WriteLine($"Generated {timesteps} audio frames");

        // Decode to waveform via vocoder
        var waveform = _vocoder.Decode(codes);

        // Write WAV file
        await Task.Run(() => WavWriter.Write(outputPath, waveform, sampleRate: 24000));

        Console.WriteLine($"Saved {outputPath} ({waveform.Length} samples, {waveform.Length / 24000.0:F2}s)");
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _embeddings.Dispose();
        _languageModel.Dispose();
        _vocoder.Dispose();
    }
}
