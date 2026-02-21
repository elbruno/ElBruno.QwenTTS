using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace QwenTTS.Models;

/// <summary>
/// ONNX language model for Qwen3-TTS.
/// Runs autoregressive inference with KV-cache to generate audio codes
/// from tokenized text input.
/// </summary>
public sealed class LanguageModel : IDisposable
{
    private readonly InferenceSession _prefillSession;
    private readonly InferenceSession _decodeSession;
    private readonly InferenceSession _cpSession;
    private readonly EmbeddingStore _embeddings;

    public LanguageModel(string modelDir, EmbeddingStore embeddings)
    {
        _embeddings = embeddings;

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        _prefillSession = new InferenceSession(Path.Combine(modelDir, "talker_prefill.onnx"), options);
        _decodeSession = new InferenceSession(Path.Combine(modelDir, "talker_decode.onnx"), options);
        _cpSession = new InferenceSession(Path.Combine(modelDir, "code_predictor.onnx"), options);
    }

    public long[,,] Generate(int[] tokenIds, string speaker, string language,
                             int maxNewTokens = 2048, float temperature = 0.9f,
                             int topK = 50, float topP = 1.0f,
                             float repetitionPenalty = 1.05f)
    {
        var cfg = _embeddings.Config;
        var speakerId = _embeddings.GetSpeakerId(speaker.ToLowerInvariant());
        
        // Build prefill embedding
        var (inputsEmbeds, trailingTextHidden) = BuildPrefillEmbedding(tokenIds, speakerId, language, cfg);
        int prefillLen = inputsEmbeds.GetLength(1);

        // Attention mask: all 1s
        var attentionMask = new long[1, prefillLen];
        for (int i = 0; i < prefillLen; i++)
            attentionMask[0, i] = 1;

        // Position IDs: cumsum(mask) - 1, broadcast to (3, 1, T)
        var positionIds = new long[3, 1, prefillLen];
        for (int ax = 0; ax < 3; ax++)
            for (int i = 0; i < prefillLen; i++)
                positionIds[ax, 0, i] = i;

        // Run prefill
        var prefillInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", new DenseTensor<float>(inputsEmbeds.Cast<float>().ToArray(), new[] { 1, prefillLen, 1024 })),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask.Cast<long>().ToArray(), new[] { 1, prefillLen })),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds.Cast<long>().ToArray(), new[] { 3, 1, prefillLen }))
        };

        using var prefillOutputs = _prefillSession.Run(prefillInputs);
        var logits = prefillOutputs.First(x => x.Name == "logits").AsEnumerable<float>().ToArray();
        var hiddenStates = prefillOutputs.First(x => x.Name == "hidden_states").AsEnumerable<float>().ToArray();

        // Stack KV cache from flat outputs
        var (pastKeys, pastValues) = StackPrefillKV(prefillOutputs, prefillLen);

        // Generation loop
        var generatedCodes = new List<long[]>();
        var generatedTokens = new List<int>();
        
        Span<float> tempEmbed2048 = stackalloc float[2048];
        Span<float> ttsEosEmbed = stackalloc float[1024];
        Span<float> ttsPadEmbed = stackalloc float[1024];
        _embeddings.TextEmbedding(cfg.tts.tts_eos_token_id, tempEmbed2048);
        _embeddings.TextProjection(tempEmbed2048, ttsEosEmbed);
        _embeddings.TextEmbedding(cfg.tts.tts_pad_token_id, tempEmbed2048);
        _embeddings.TextProjection(tempEmbed2048, ttsPadEmbed);

        // Reusable buffers for the generation loop
        var group0EmbedBuf = new float[1024];
        var cpEmbedBuf = new float[1024];
        var nextInputBuf = new float[1024];

        for (int step = 0; step < maxNewTokens; step++)
        {
            // Sample group 0
            var group0Token = SampleToken(logits, temperature, topK, topP, repetitionPenalty, generatedTokens, cfg);
            if (group0Token == cfg.talker.codec_eos_token_id)
                break;

            generatedTokens.Add(group0Token);

            // Generate groups 1-15 via Code Predictor
            var codes = new long[16];
            codes[0] = group0Token;
            
            _embeddings.TalkerCodecEmbedding(group0Token, group0EmbedBuf);

            // CP prefill: concat(hidden_states[:,-1,:], group0_embed)
            var cpInputs = new float[1, 2, 1024];
            int hOffset = hiddenStates.Length - 1024; // last hidden state position
            for (int i = 0; i < 1024; i++)
            {
                cpInputs[0, 0, i] = hiddenStates[hOffset + i];
                cpInputs[0, 1, i] = group0EmbedBuf[i];
            }

            var (cpPastKeys, cpPastValues) = InitCPKV();
            int cpPastLen = 0;

            for (int groupIdx = 1; groupIdx < 16; groupIdx++)
            {
                int cpInputSeqLen = groupIdx == 1 ? 2 : 1;
                var cpInputsList = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("inputs_embeds", new DenseTensor<float>(cpInputs.Cast<float>().ToArray(), new[] { 1, cpInputSeqLen, 1024 })),
                    NamedOnnxValue.CreateFromTensor("generation_steps", new DenseTensor<long>(new[] { (long)(groupIdx - 1) }, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(cpPastKeys, new[] { 5, 1, 8, cpPastLen, 128 })),
                    NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(cpPastValues, new[] { 5, 1, 8, cpPastLen, 128 }))
                };

                using var cpOutputs = _cpSession.Run(cpInputsList);
                var cpLogits = cpOutputs.First(x => x.Name == "logits").AsEnumerable<float>().ToArray();
                var cpToken = SampleTokenSimple(cpLogits, temperature);
                codes[groupIdx] = cpToken;

                // Update CP KV
                cpPastLen += cpInputSeqLen;
                cpPastKeys = cpOutputs.First(x => x.Name == "present_keys").AsEnumerable<float>().ToArray();
                cpPastValues = cpOutputs.First(x => x.Name == "present_values").AsEnumerable<float>().ToArray();

                // Next input
                if (groupIdx < 15)
                {
                    _embeddings.CpCodecEmbedding(groupIdx - 1, cpToken, cpEmbedBuf);
                    cpInputs = new float[1, 1, 1024];
                    for (int i = 0; i < 1024; i++)
                        cpInputs[0, 0, i] = cpEmbedBuf[i];
                }
            }

            generatedCodes.Add(codes);

            // Build next Talker input: sum all group embeddings + trailing text
            _embeddings.TalkerCodecEmbedding((int)codes[0], nextInputBuf);
            for (int g = 1; g < 16; g++)
            {
                _embeddings.CpCodecEmbedding(g - 1, (int)codes[g], cpEmbedBuf);
                for (int i = 0; i < 1024; i++)
                    nextInputBuf[i] += cpEmbedBuf[i];
            }

            // Add trailing text hidden or tts_pad
            if (step < trailingTextHidden.GetLength(0))
            {
                for (int i = 0; i < 1024; i++)
                    nextInputBuf[i] += trailingTextHidden[step, i];
            }
            else
            {
                for (int i = 0; i < 1024; i++)
                    nextInputBuf[i] += ttsPadEmbed[i];
            }

            // Update attention mask and position IDs
            int newLen = prefillLen + step + 1;
            var newAttentionMask = new long[1, newLen];
            for (int i = 0; i < newLen; i++)
                newAttentionMask[0, i] = 1;

            var newPositionIds = new long[3, 1, 1];
            for (int ax = 0; ax < 3; ax++)
                newPositionIds[ax, 0, 0] = prefillLen + step;

            // Run decode
            var decodeInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds", new DenseTensor<float>(nextInputBuf, new[] { 1, 1, 1024 })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(newAttentionMask.Cast<long>().ToArray(), new[] { 1, newLen })),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(newPositionIds.Cast<long>().ToArray(), new[] { 3, 1, 1 })),
                NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, new[] { 28, 1, 8, prefillLen + step, 128 })),
                NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, new[] { 28, 1, 8, prefillLen + step, 128 }))
            };

            using var decodeOutputs = _decodeSession.Run(decodeInputs);
            logits = decodeOutputs.First(x => x.Name == "logits").AsEnumerable<float>().ToArray();
            hiddenStates = decodeOutputs.First(x => x.Name == "hidden_states").AsEnumerable<float>().ToArray();
            pastKeys = decodeOutputs.First(x => x.Name == "present_keys").AsEnumerable<float>().ToArray();
            pastValues = decodeOutputs.First(x => x.Name == "present_values").AsEnumerable<float>().ToArray();
        }

        // Convert to (1, 16, T)
        int T = generatedCodes.Count;
        var result = new long[1, 16, T];
        for (int t = 0; t < T; t++)
            for (int g = 0; g < 16; g++)
                result[0, g, t] = generatedCodes[t][g];

        return result;
    }

    private (float[,,], float[,]) BuildPrefillEmbedding(int[] tokenIds, int speakerId, string language, ModelConfig cfg)
    {
        // Role embed: tokens [0:3]
        var roleEmbeds = new List<float[]>();
        var roleTextEmbBuf = new float[2048];
        var roleProjBuf = new float[1024];
        for (int i = 0; i < 3; i++)
        {
            _embeddings.TextEmbedding(tokenIds[i], roleTextEmbBuf);
            _embeddings.TextProjection(roleTextEmbBuf, roleProjBuf);
            roleEmbeds.Add((float[])roleProjBuf.Clone());
        }

        // Codec prefix
        var codecPrefix = new List<int>();
        if (language != "auto")
        {
            codecPrefix.Add(cfg.talker.codec_think_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            codecPrefix.Add(cfg.language_ids[language.ToLowerInvariant()]);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }
        else
        {
            codecPrefix.Add(cfg.talker.codec_nothink_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }
        
        codecPrefix.Add(speakerId);
        codecPrefix.Add(cfg.talker.codec_pad_id);
        codecPrefix.Add(cfg.talker.codec_bos_id);

        // TTS token embeds
        Span<float> ttsPadTextEmbed = stackalloc float[2048];
        Span<float> ttsBosTextEmbed = stackalloc float[2048];
        Span<float> ttsEosTextEmbed = stackalloc float[2048];
        Span<float> ttsPadProj = stackalloc float[1024];
        Span<float> ttsBosProj = stackalloc float[1024];
        Span<float> ttsEosProj = stackalloc float[1024];
        
        _embeddings.TextEmbedding(cfg.tts.tts_pad_token_id, ttsPadTextEmbed);
        _embeddings.TextProjection(ttsPadTextEmbed, ttsPadProj);
        _embeddings.TextEmbedding(cfg.tts.tts_bos_token_id, ttsBosTextEmbed);
        _embeddings.TextProjection(ttsBosTextEmbed, ttsBosProj);
        _embeddings.TextEmbedding(cfg.tts.tts_eos_token_id, ttsEosTextEmbed);
        _embeddings.TextProjection(ttsEosTextEmbed, ttsEosProj);

        // _talker_input_embed = concat(tts_pad.expand(codec_prefix_len - 2), tts_bos) + codec_prefix[:-1]
        var talkerInputEmbeds = new List<float[]>();
        int codecPrefixLen = codecPrefix.Count;
        var codecEmbBuf = new float[1024];
        
        for (int i = 0; i < codecPrefixLen - 2; i++)
        {
            _embeddings.TalkerCodecEmbedding(codecPrefix[i], codecEmbBuf);
            var combined = new float[1024];
            for (int j = 0; j < 1024; j++)
                combined[j] = ttsPadProj[j] + codecEmbBuf[j];
            talkerInputEmbeds.Add(combined);
        }
        
        // Last position: tts_bos + codec_prefix[-2]
        {
            _embeddings.TalkerCodecEmbedding(codecPrefix[codecPrefixLen - 2], codecEmbBuf);
            var combined = new float[1024];
            for (int j = 0; j < 1024; j++)
                combined[j] = ttsBosProj[j] + codecEmbBuf[j];
            talkerInputEmbeds.Add(combined);
        }

        // Combine: role_embed + _talker_input_embed
        var allEmbeds = new List<float[]>();
        allEmbeds.AddRange(roleEmbeds);
        allEmbeds.AddRange(talkerInputEmbeds);

        // Append: TextProject(TextEmbed(tokenIds[3])) + codec_bos
        {
            Span<float> textEmbed = stackalloc float[2048];
            Span<float> projected = stackalloc float[1024];
            _embeddings.TextEmbedding(tokenIds[3], textEmbed);
            _embeddings.TextProjection(textEmbed, projected);
            
            Span<float> codecBosEmbed = stackalloc float[1024];
            _embeddings.TalkerCodecEmbedding(cfg.talker.codec_bos_id, codecBosEmbed);
            
            var combined = new float[1024];
            for (int j = 0; j < 1024; j++)
                combined[j] = projected[j] + codecBosEmbed[j];
            allEmbeds.Add(combined);
        }

        // Trailing text hidden: tokens[4:-5] + tts_eos
        var trailingList = new List<float[]>();
        var trailTextEmbBuf = new float[2048];
        var trailProjBuf = new float[1024];
        for (int i = 4; i < tokenIds.Length - 5; i++)
        {
            _embeddings.TextEmbedding(tokenIds[i], trailTextEmbBuf);
            _embeddings.TextProjection(trailTextEmbBuf, trailProjBuf);
            trailingList.Add((float[])trailProjBuf.Clone());
        }
        trailingList.Add(ttsEosProj.ToArray());

        // Convert to arrays
        var inputsEmbeds = new float[1, allEmbeds.Count, 1024];
        for (int i = 0; i < allEmbeds.Count; i++)
            for (int j = 0; j < 1024; j++)
                inputsEmbeds[0, i, j] = allEmbeds[i][j];

        var trailingTextHidden = new float[trailingList.Count, 1024];
        for (int i = 0; i < trailingList.Count; i++)
            for (int j = 0; j < 1024; j++)
                trailingTextHidden[i, j] = trailingList[i][j];

        return (inputsEmbeds, trailingTextHidden);
    }

    private (float[], float[]) StackPrefillKV(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, int seqLen)
    {
        // 56 tensors: present_key_0..27, present_value_0..27 each (1, 8, T, 128)
        var keys = new float[28 * 1 * 8 * seqLen * 128];
        var values = new float[28 * 1 * 8 * seqLen * 128];

        for (int layer = 0; layer < 28; layer++)
        {
            var keyData = outputs.First(x => x.Name == $"present_key_{layer}").AsEnumerable<float>().ToArray();
            var valueData = outputs.First(x => x.Name == $"present_value_{layer}").AsEnumerable<float>().ToArray();
            
            Array.Copy(keyData, 0, keys, layer * 1 * 8 * seqLen * 128, keyData.Length);
            Array.Copy(valueData, 0, values, layer * 1 * 8 * seqLen * 128, valueData.Length);
        }

        return (keys, values);
    }

    private (float[], float[]) InitCPKV()
    {
        // Empty KV for CP prefill: (5, 1, 8, 0, 128)
        return (Array.Empty<float>(), Array.Empty<float>());
    }

    private int SampleToken(float[] logits, float temperature, int topK, float topP, float repPenalty,
                            List<int> previousTokens, ModelConfig cfg)
    {
        // Logits are (1, 1, 3072), flatten to (3072,)
        var vocabSize = cfg.talker.vocab_size;
        var probs = new float[vocabSize];
        Array.Copy(logits, logits.Length - vocabSize, probs, 0, vocabSize);

        // Apply repetition penalty
        foreach (var token in previousTokens)
            probs[token] /= repPenalty;

        // Suppress tokens [2048, 3071] except codec_eos_token_id
        for (int i = vocabSize - 1024; i < vocabSize; i++)
        {
            if (i != cfg.talker.codec_eos_token_id)
                probs[i] = float.NegativeInfinity;
        }

        // Temperature
        if (temperature > 0)
        {
            for (int i = 0; i < vocabSize; i++)
                probs[i] /= temperature;
        }

        // Top-k
        if (topK > 0 && topK < vocabSize)
        {
            var indexed = probs.Select((p, i) => (p, i)).OrderByDescending(x => x.p).ToArray();
            for (int i = topK; i < vocabSize; i++)
                probs[indexed[i].i] = float.NegativeInfinity;
        }

        // Softmax
        float maxLogit = probs.Max();
        float sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = MathF.Exp(probs[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++)
            probs[i] /= sumExp;

        // Multinomial sample
        float rand = Random.Shared.NextSingle();
        float cumSum = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumSum += probs[i];
            if (rand < cumSum)
                return i;
        }

        return vocabSize - 1;
    }

    private int SampleTokenSimple(float[] logits, float temperature)
    {
        // Logits are (1, 1, 2048)
        var vocabSize = 2048;
        var probs = new float[vocabSize];
        Array.Copy(logits, logits.Length - vocabSize, probs, 0, vocabSize);

        if (temperature > 0)
        {
            for (int i = 0; i < vocabSize; i++)
                probs[i] /= temperature;
        }

        // Softmax
        float maxLogit = probs.Max();
        float sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = MathF.Exp(probs[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++)
            probs[i] /= sumExp;

        // Sample
        float rand = Random.Shared.NextSingle();
        float cumSum = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumSum += probs[i];
            if (rand < cumSum)
                return i;
        }

        return vocabSize - 1;
    }

    public void Dispose()
    {
        _prefillSession.Dispose();
        _decodeSession.Dispose();
        _cpSession.Dispose();
    }
}
