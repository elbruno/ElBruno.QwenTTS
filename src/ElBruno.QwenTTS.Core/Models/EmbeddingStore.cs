using System.Text.Json;

namespace ElBruno.QwenTTS.Models;

/// <summary>
/// Loads and provides access to all embedding matrices (.npy files).
/// Includes text embeddings, projection layers, codec embeddings, and speaker IDs.
/// </summary>
internal sealed class EmbeddingStore : IDisposable
{
    private readonly float[,] _textEmbedding;           // (vocab_size, 2048)
    private readonly float[,] _fc1Weight;               // (2048, 2048)
    private readonly float[] _fc1Bias;                  // (2048,)
    private readonly float[,] _fc2Weight;               // (1024, 2048)
    private readonly float[] _fc2Bias;                  // (1024,)
    private readonly float[,] _talkerCodecEmbedding;    // (3072, 1024)
    private readonly float[][,] _cpCodecEmbeddings;     // 15 × (2048, 1024)
    private readonly Dictionary<string, int> _speakerIds;
    
    public ModelConfig Config { get; }

    public EmbeddingStore(string embeddingsDir, string configPath)
    {
        // Load config
        var configJson = File.ReadAllText(configPath);
        Config = JsonSerializer.Deserialize<ModelConfig>(configJson)
            ?? throw new InvalidDataException("Failed to parse config.json");

        // Load text embedding and projection
        _textEmbedding = NpyReader.ReadFloat2D(Path.Combine(embeddingsDir, "text_embedding.npy"));
        _fc1Weight = NpyReader.ReadFloat2D(Path.Combine(embeddingsDir, "text_projection_fc1_weight.npy"));
        _fc1Bias = NpyReader.ReadFloat1D(Path.Combine(embeddingsDir, "text_projection_fc1_bias.npy"));
        _fc2Weight = NpyReader.ReadFloat2D(Path.Combine(embeddingsDir, "text_projection_fc2_weight.npy"));
        _fc2Bias = NpyReader.ReadFloat1D(Path.Combine(embeddingsDir, "text_projection_fc2_bias.npy"));

        // Load talker codec embedding
        _talkerCodecEmbedding = NpyReader.ReadFloat2D(Path.Combine(embeddingsDir, "talker_codec_embedding.npy"));

        // Load CP codec embeddings (15 groups)
        _cpCodecEmbeddings = new float[15][,];
        for (int i = 0; i < 15; i++)
        {
            var path = Path.Combine(embeddingsDir, $"cp_codec_embedding_{i}.npy");
            _cpCodecEmbeddings[i] = NpyReader.ReadFloat2D(path);
        }

        // Load speaker IDs
        var speakerIdsPath = Path.Combine(embeddingsDir, "speaker_ids.json");
        var speakerJson = File.ReadAllText(speakerIdsPath);
        _speakerIds = JsonSerializer.Deserialize<Dictionary<string, int>>(speakerJson)
            ?? throw new InvalidDataException("Failed to parse speaker_ids.json");
    }

    /// <summary>
    /// Looks up text embedding for a token ID and writes to output.
    /// </summary>
    public void TextEmbedding(int tokenId, Span<float> output)
    {
        if (output.Length != 2048)
            throw new ArgumentException("Output must be length 2048");
        
        for (int i = 0; i < 2048; i++)
            output[i] = _textEmbedding[tokenId, i];
    }

    /// <summary>
    /// Applies text projection MLP: output = fc2(silu(fc1(input)))
    /// </summary>
    public void TextProjection(ReadOnlySpan<float> input2048, Span<float> output1024)
    {
        if (input2048.Length != 2048)
            throw new ArgumentException("Input must be length 2048");
        if (output1024.Length != 1024)
            throw new ArgumentException("Output must be length 1024");

        // fc1: (2048, 2048) @ input + bias → hidden (2048,)
        Span<float> hidden = stackalloc float[2048];
        MatMul(_fc1Weight, input2048, hidden);
        for (int i = 0; i < 2048; i++)
            hidden[i] = SiLU(hidden[i] + _fc1Bias[i]);

        // fc2: (1024, 2048) @ hidden + bias → output (1024,)
        MatMul(_fc2Weight, hidden, output1024);
        for (int i = 0; i < 1024; i++)
            output1024[i] += _fc2Bias[i];
    }

    /// <summary>
    /// Looks up talker codec embedding for a token ID.
    /// </summary>
    public void TalkerCodecEmbedding(int tokenId, Span<float> output1024)
    {
        if (output1024.Length != 1024)
            throw new ArgumentException("Output must be length 1024");
        
        for (int i = 0; i < 1024; i++)
            output1024[i] = _talkerCodecEmbedding[tokenId, i];
    }

    /// <summary>
    /// Looks up CP codec embedding for a group and token ID.
    /// groupIndex is 0-14 (maps to cp_codec_embedding_0..14).
    /// </summary>
    public void CpCodecEmbedding(int groupIndex, int tokenId, Span<float> output1024)
    {
        if (groupIndex < 0 || groupIndex >= 15)
            throw new ArgumentException($"groupIndex must be 0-14, got {groupIndex}");
        if (output1024.Length != 1024)
            throw new ArgumentException("Output must be length 1024");
        
        var table = _cpCodecEmbeddings[groupIndex];
        for (int i = 0; i < 1024; i++)
            output1024[i] = table[tokenId, i];
    }

    /// <summary>
    /// Gets the speaker token ID for a speaker name.
    /// </summary>
    public int GetSpeakerId(string speaker)
    {
        if (!_speakerIds.TryGetValue(speaker, out var id))
            throw new ArgumentException($"Unknown speaker: {speaker}");
        return id;
    }

    /// <summary>
    /// Gets the list of available speaker names.
    /// </summary>
    public IReadOnlyCollection<string> GetAvailableSpeakers() => _speakerIds.Keys;

    /// <summary>
    /// Gets the talker codec embedding for a speaker as a float array.
    /// Used for similarity search and speaker matching.
    /// </summary>
    /// <param name="speakerId">Speaker token ID (codec embedding token).</param>
    /// <returns>Embedding vector (1024 dimensions).</returns>
    public float[] GetSpeakerEmbedding(int speakerId)
    {
        var embedding = new float[1024];
        for (int i = 0; i < 1024; i++)
            embedding[i] = _talkerCodecEmbedding[speakerId, i];
        return embedding;
    }

    /// <summary>
    /// Gets all speaker embeddings as a collection for similarity search.
    /// </summary>
    /// <returns>Tuples of (speakerName, embedding) for all available speakers.</returns>
    public IEnumerable<(string name, float[] embedding)> GetAllSpeakerEmbeddings()
    {
        foreach (var (name, id) in _speakerIds)
        {
            yield return (name, GetSpeakerEmbedding(id));
        }
    }

    public void Dispose()
    {
        // No unmanaged resources
    }

    private static float SiLU(float x) => x / (1.0f + MathF.Exp(-x));

    /// <summary>
    /// Matrix-vector multiply: output = weight @ input
    /// weight is (M, N), input is (N,), output is (M,)
    /// </summary>
    private static void MatMul(float[,] weight, ReadOnlySpan<float> input, Span<float> output)
    {
        int M = weight.GetLength(0);
        int N = weight.GetLength(1);
        
        for (int i = 0; i < M; i++)
        {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += weight[i, j] * input[j];
            output[i] = sum;
        }
    }
}

/// <summary>
/// Model configuration loaded from config.json
/// </summary>
internal sealed class ModelConfig
{
    public TalkerConfig talker { get; set; } = new();
    public CodePredictorConfig code_predictor { get; set; } = new();
    public TtsConfig tts { get; set; } = new();
    public Dictionary<string, int> language_ids { get; set; } = new();
    public Dictionary<string, object> speaker_dialect { get; set; } = new();
}

internal sealed class TalkerConfig
{
    public int codec_eos_token_id { get; set; }
    public int codec_pad_id { get; set; }
    public int codec_bos_id { get; set; }
    public int codec_think_id { get; set; }
    public int codec_nothink_id { get; set; }
    public int codec_think_bos_id { get; set; }
    public int codec_think_eos_id { get; set; }
    public int num_code_groups { get; set; }
    public int hidden_size { get; set; }
    public int text_hidden_size { get; set; }
    public int num_hidden_layers { get; set; }
    public int num_key_value_heads { get; set; }
    public int head_dim { get; set; }
    public int vocab_size { get; set; }
}

internal sealed class CodePredictorConfig
{
    public int num_hidden_layers { get; set; }
    public int num_key_value_heads { get; set; }
    public int head_dim { get; set; }
    public int vocab_size { get; set; }
}

internal sealed class TtsConfig
{
    public int tts_bos_token_id { get; set; }
    public int tts_eos_token_id { get; set; }
    public int tts_pad_token_id { get; set; }
}
