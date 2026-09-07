using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ElBruno.QwenTTS.Models;

/// <summary>
/// Loads and runs the vocoder ONNX model.
/// Converts discrete audio codes (16 RVQ codebooks) to PCM waveform at 24 kHz.
/// Input shape: (1, 16, T) int64 — Output shape: (1, 1, T*1920) float32.
/// </summary>
internal sealed class Vocoder : IDisposable
{
    private readonly string _modelPath;
    private readonly VocoderSessionRunner<VocoderSessionState> _sessionRunner;

    public int SampleRate => 24000;

    /// <summary>Number of PCM samples per code frame (24000 Hz / 12 Hz = 1920).</summary>
    public const int SamplesPerFrame = 1920;

    /// <summary>
    /// Creates a vocoder wrapper. Session is loaded lazily on first Decode call.
    /// </summary>
    /// <param name="modelPath">Path to the vocoder ONNX model.</param>
    /// <param name="sessionOptionsFactory">Optional factory for ONNX Runtime session options (e.g., for GPU acceleration).</param>
    public Vocoder(string modelPath, Func<SessionOptions>? sessionOptionsFactory = null)
    {
        _modelPath = modelPath;
        _sessionRunner = new(
            () => CreateSessionState(sessionOptionsFactory ?? CreateDefaultOptions),
            sessionOptionsFactory is null ? null : () => CreateSessionState(CreateDefaultOptions));
    }

    private static SessionOptions CreateDefaultOptions() => new()
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    };

    private VocoderSessionState CreateSessionState(Func<SessionOptions> sessionOptionsFactory)
    {
        // SEC-3: File size pre-check to prevent out-of-memory attacks
        // Raised to 8 GB for consistency with LanguageModel.cs (1.7B model support)
        var fileInfo = new FileInfo(_modelPath);
        const long maxOnnxSize = 8_000_000_000; // 8 GB
        if (fileInfo.Length > maxOnnxSize)
            throw new InvalidOperationException($"ONNX file too large ({fileInfo.Length / 1e9:F2} GB). Maximum allowed: {maxOnnxSize / 1e9:F2} GB.");

        using var options = sessionOptionsFactory();
        var session = new InferenceSession(_modelPath, options);
        var inputName = session.InputMetadata.Keys.FirstOrDefault() ?? "codes";
        return new VocoderSessionState(session, inputName);
    }

    /// <summary>
    /// Decode audio codes to waveform.
    /// </summary>
    /// <param name="codes">Audio codes of shape (1, 16, T) where T is the number of timesteps.</param>
    /// <param name="cancellationToken">Cancellation token checked between flattening, inference, and output copy phases.</param>
    /// <returns>PCM waveform samples at 24 kHz, values in [-1, 1].</returns>
    public float[] Decode(long[,,] codes, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int batch = codes.GetLength(0);      // 1
        int quantizers = codes.GetLength(1);  // 16
        int timesteps = codes.GetLength(2);   // T

        // Flatten 3D array into DenseTensor (row-major)
        var tensor = new DenseTensor<long>(new[] { batch, quantizers, timesteps });
        for (int b = 0; b < batch; b++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int q = 0; q < quantizers; q++)
                for (int t = 0; t < timesteps; t++)
                    tensor[b, q, t] = codes[b, q, t];
        }

        using var results = _sessionRunner.Run(sessionState =>
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(sessionState.InputName, tensor)
            };
            return sessionState.Session.Run(inputs);
        }, cancellationToken);

        // Extract waveform from first output tensor
        var outputTensor = results.First().AsTensor<float>();

        // Validate output size matches expected upsample factor (12 Hz codes → 24 kHz PCM = 1920×)
        int expectedSamples = timesteps * SamplesPerFrame;
        if (outputTensor.Length != expectedSamples)
        {
            throw new InvalidOperationException(
                $"Vocoder output mismatch: expected {expectedSamples} samples " +
                $"({timesteps} frames × {SamplesPerFrame}), got {outputTensor.Length}. " +
                $"The vocoder ONNX model may have been exported with a fixed time dimension. " +
                $"Re-export with dynamic_axes on the timesteps dimension.");
        }

        // Copy tensor values to a flat array
        var waveform = new float[outputTensor.Length];
        int i = 0;
        foreach (var sample in outputTensor)
        {
            if ((i & 0xFFF) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            waveform[i++] = sample;
        }

        return waveform;
    }

    public void Dispose() => _sessionRunner.Dispose();

    private sealed record VocoderSessionState(InferenceSession Session, string InputName) : IDisposable
    {
        public void Dispose() => Session.Dispose();
    }
}
