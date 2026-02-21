using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace QwenTTS.Models;

/// <summary>
/// GPT-2 style BPE tokenizer for Qwen3-TTS.
/// Loads vocab.json and merges.txt to produce token IDs identical to
/// the Python HuggingFace tokenizer (Qwen2Tokenizer).
/// </summary>
public sealed class TextTokenizer : IDisposable
{
    // --- Special token IDs (from model config / tokenizer_config.json) ---
    public const int EndOfTextId = 151643;
    public const int ImStartId = 151644;
    public const int ImEndId = 151645;
    public const int AudioStartId = 151669;
    public const int AudioEndId = 151670;
    public const int TtsPadId = 151671;
    public const int TtsBosId = 151672;
    public const int TtsEodId = 151673;
    public const int TtsBosSignleId = 151674;
    public const int AudioPadId = 151675;

    // Well-known token IDs referenced by the chat template
    public const int AssistantTokenId = 77091;
    public const int NewlineTokenId = 198;

    private static readonly IReadOnlyDictionary<string, int> SpecialTokensMap =
        new Dictionary<string, int>
        {
            ["<|endoftext|>"] = EndOfTextId,
            ["<|im_start|>"] = ImStartId,
            ["<|im_end|>"] = ImEndId,
            ["<|audio_start|>"] = AudioStartId,
            ["<|audio_end|>"] = AudioEndId,
            ["<tts_pad>"] = TtsPadId,
            ["<tts_text_bos>"] = TtsBosId,
            ["<tts_text_eod>"] = TtsEodId,
            ["<tts_text_bos_single>"] = TtsBosSignleId,
            ["<|audio_pad|>"] = AudioPadId,
        };

    // GPT-2 byte-level BPE pre-tokenization pattern
    private static readonly Regex Gpt2Regex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    private readonly BpeTokenizer _tokenizer;

    /// <summary>
    /// Loads the BPE tokenizer from vocab.json and merges.txt in <paramref name="modelDir"/>.
    /// </summary>
    public TextTokenizer(string modelDir)
    {
        var vocabPath = Path.Combine(modelDir, "vocab.json");
        var mergesPath = Path.Combine(modelDir, "merges.txt");

        if (!File.Exists(vocabPath))
            throw new FileNotFoundException("vocab.json not found in model directory.", vocabPath);
        if (!File.Exists(mergesPath))
            throw new FileNotFoundException("merges.txt not found in model directory.", mergesPath);

        var preTokenizer = new RegexPreTokenizer(Gpt2Regex, SpecialTokensMap);

        var options = new BpeOptions(vocabPath, mergesPath)
        {
            PreTokenizer = preTokenizer,
            SpecialTokens = SpecialTokensMap,
            ByteLevel = true,
            FuseUnknownTokens = false,
        };

        _tokenizer = BpeTokenizer.Create(options);
    }

    /// <summary>
    /// Encodes text into BPE token IDs. Handles special tokens in the input
    /// (e.g. &lt;|im_start|&gt;) via the pre-tokenizer.
    /// </summary>
    public int[] Encode(string text)
    {
        var ids = _tokenizer.EncodeToIds(text,
            considerPreTokenization: true,
            considerNormalization: false);
        return [.. ids];
    }

    /// <summary>
    /// Decodes token IDs back to text (for debugging / verification).
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        return _tokenizer.Decode(tokenIds, considerSpecialTokens: true) ?? string.Empty;
    }

    /// <summary>
    /// Builds the full Qwen3-TTS CustomVoice prompt token sequence.
    /// Wraps the text in the chat template and optionally prepends an instruct turn.
    /// </summary>
    /// <remarks>
    /// Output token layout (without instruct):
    ///   [im_start, assistant, \n, ...text tokens..., im_end, \n, im_start, assistant, \n]
    ///
    /// With instruct prepended:
    ///   [im_start, user, \n, ...instruct tokens..., im_end, \n, ...main tokens above...]
    ///
    /// The caller (embedding builder) splits this into role / text / suffix per TOKENIZER.MD §4.
    /// </remarks>
    public int[] BuildCustomVoicePrompt(string text, string speaker, string language, string? instruct = null)
    {
        var tokens = new List<int>();

        // Instruct turn (user message) — only used by larger models
        if (!string.IsNullOrEmpty(instruct))
        {
            var instructWrapped = $"<|im_start|>user\n{instruct}<|im_end|>\n";
            tokens.AddRange(Encode(instructWrapped));
        }

        // Main assistant turn with trailing continuation marker
        var assistantWrapped = $"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n";
        tokens.AddRange(Encode(assistantWrapped));

        return [.. tokens];
    }

    public void Dispose()
    {
        // BpeTokenizer does not hold unmanaged resources
    }
}
