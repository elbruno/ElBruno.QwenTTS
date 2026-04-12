using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ElBruno.QwenTTS.VoiceCloning.Models;

/// <summary>
/// Wraps the speech tokenizer encoder ONNX model.
/// Encodes raw 24 kHz audio into quantized codec codes [1, T, 16].
/// Used in ICL (In-Context Learning) mode to extract reference audio codes
/// for higher-quality voice cloning.
/// </summary>
internal sealed class SpeechTokenizer : IDisposable
{
    private InferenceSession? _session;
    private readonly string _modelPath;
    private readonly Func<SessionOptions> _sessionOptionsFactory;

    /// <summary>Expected sample rate for input audio.</summary>
    public const int SampleRate = 24000;

    /// <summary>Number of samples per codec frame.</summary>
    public const int SamplesPerFrame = 1920;

    /// <summary>Number of codebook groups in the output.</summary>
    public const int NumCodebooks = 16;

    public SpeechTokenizer(string modelPath, Func<SessionOptions>? sessionOptionsFactory = null)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Speech tokenizer model not found.", modelPath);

        _modelPath = modelPath;
        _sessionOptionsFactory = sessionOptionsFactory ?? CreateDefaultOptions;
    }

    private static SessionOptions CreateDefaultOptions() => new()
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    };

    private InferenceSession GetSession()
    {
        return _session ??= new InferenceSession(_modelPath, _sessionOptionsFactory());
    }

    /// <summary>
    /// Encode raw audio samples into quantized codec codes.
    /// </summary>
    /// <param name="samples">PCM float32 samples at 24 kHz, mono.</param>
    /// <returns>Codec codes of shape [1, T_frames, 16] where T_frames = ceil(num_samples / 1920).</returns>
    public long[,,] Encode(float[] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0)
            throw new ArgumentException("Audio samples cannot be empty.", nameof(samples));

        // ONNX model expects input shape [B, 1, T_samples] (3D) named "audio_waveform"
        var inputTensor = new DenseTensor<float>(samples, [1, 1, samples.Length]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_waveform", inputTensor)
        };

        using var results = GetSession().Run(inputs);
        var outputTensor = results.First().AsTensor<long>();

        // ONNX output shape: [B, 16, T_frames] — transpose to [B, T_frames, 16]
        var dims = outputTensor.Dimensions;
        int batch = dims[0];
        int codebooks = dims[1];
        int tFrames = dims[2];

        var codes = new long[batch, tFrames, codebooks];
        int idx = 0;
        foreach (var val in outputTensor)
        {
            int b = idx / (codebooks * tFrames);
            int cb = (idx % (codebooks * tFrames)) / tFrames;
            int t = idx % tFrames;
            codes[b, t, cb] = val;
            idx++;
        }

        return codes;
    }

    /// <summary>
    /// Encode a WAV file into quantized codec codes.
    /// Reads the WAV file, resamples to 24 kHz if needed, and encodes.
    /// </summary>
    /// <param name="wavPath">Path to a WAV file.</param>
    /// <returns>Codec codes of shape [1, T_frames, 16].</returns>
    public long[,,] EncodeFromWav(string wavPath)
    {
        var (samples, sampleRate) = Audio.MelSpectrogram.ReadWav(wavPath);
        if (sampleRate != SampleRate)
            samples = Audio.MelSpectrogram.Resample(samples, sampleRate, SampleRate);
        return Encode(samples);
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
