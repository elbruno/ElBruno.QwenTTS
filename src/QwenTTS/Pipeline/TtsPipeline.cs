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

    /// <summary>
    /// Creates a TTS pipeline from the model directory containing
    /// tokenizer files, LM ONNX model, and vocoder ONNX model.
    /// </summary>
    public TtsPipeline(string modelDir)
    {
        // TODO: Resolve actual file paths within modelDir
        _tokenizer = new TextTokenizer(modelDir);
        _languageModel = new LanguageModel(Path.Combine(modelDir, "lm.onnx"));
        _vocoder = new Vocoder(Path.Combine(modelDir, "vocoder.onnx"));
    }

    /// <summary>
    /// Runs the full synthesis pipeline and writes the result to a WAV file.
    /// </summary>
    /// <param name="text">Text to synthesize.</param>
    /// <param name="speaker">Speaker name (e.g., Ryan).</param>
    /// <param name="outputPath">Output WAV file path.</param>
    /// <param name="instruct">Optional instruction for voice style (e.g., "speak happily").</param>
    public async Task SynthesizeAsync(string text, string speaker, string outputPath, string? instruct = null)
    {
        // TODO: Build the full prompt including speaker tag and optional instruct tag
        // Format: <speaker>{speaker}</speaker> [<instruct>{instruct}</instruct>] {text}
        var prompt = BuildPrompt(text, speaker, instruct);

        // Step 1: Tokenize
        var tokenIds = _tokenizer.Encode(prompt);

        // Step 2: Generate audio codes via LM → returns flat int[], will be reshaped
        var audioCodes = _languageModel.Generate(tokenIds);

        // Step 3: Reshape codes to (1, 16, T) and decode to waveform via vocoder
        // TODO: LanguageModel.Generate should return properly shaped codes once implemented
        int quantizers = 16;
        int timesteps = audioCodes.Length / quantizers;
        var codes = new long[1, quantizers, timesteps];
        for (int q = 0; q < quantizers; q++)
            for (int t = 0; t < timesteps; t++)
                codes[0, q, t] = audioCodes[q * timesteps + t];

        var waveform = _vocoder.Decode(codes);

        // Step 4: Write WAV file
        await Task.Run(() => WavWriter.Write(outputPath, waveform, sampleRate: 24000));

        Console.WriteLine($"Saved {outputPath} ({waveform.Length} samples, {waveform.Length / 24000.0:F2}s)");
    }

    private static string BuildPrompt(string text, string speaker, string? instruct)
    {
        // TODO: Match exact prompt format required by Qwen3-TTS
        var prompt = $"<speaker>{speaker}</speaker>";
        if (instruct is not null)
            prompt += $"<instruct>{instruct}</instruct>";
        prompt += text;
        return prompt;
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _languageModel.Dispose();
        _vocoder.Dispose();
    }
}
