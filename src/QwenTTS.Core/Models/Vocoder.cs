using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace QwenTTS.Models;

/// <summary>
/// Loads and runs the vocoder ONNX model.
/// Converts discrete audio codes (16 RVQ codebooks) to PCM waveform at 24 kHz.
/// Input shape: (1, 16, T) int64 — Output shape: (1, 1, T*1920) float32.
/// </summary>
public sealed class Vocoder : IDisposable
{
    private InferenceSession? _session;
    private string? _inputName;
    private readonly string _modelPath;

    public int SampleRate => 24000;

    /// <summary>
    /// Creates a vocoder wrapper. Session is loaded lazily on first Decode call.
    /// </summary>
    public Vocoder(string modelPath)
    {
        _modelPath = modelPath;
    }

    private InferenceSession GetSession()
    {
        if (_session is null)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };
            _session = new InferenceSession(_modelPath, options);
            _inputName = _session.InputMetadata.Keys.FirstOrDefault() ?? "codes";
        }
        return _session;
    }

    /// <summary>
    /// Decode audio codes to waveform.
    /// </summary>
    /// <param name="codes">Audio codes of shape (1, 16, T) where T is the number of timesteps.</param>
    /// <returns>PCM waveform samples at 24 kHz, values in [-1, 1].</returns>
    public float[] Decode(long[,,] codes)
    {
        int batch = codes.GetLength(0);      // 1
        int quantizers = codes.GetLength(1);  // 16
        int timesteps = codes.GetLength(2);   // T

        // Flatten 3D array into DenseTensor (row-major)
        var tensor = new DenseTensor<long>(new[] { batch, quantizers, timesteps });
        for (int b = 0; b < batch; b++)
            for (int q = 0; q < quantizers; q++)
                for (int t = 0; t < timesteps; t++)
                    tensor[b, q, t] = codes[b, q, t];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName ?? "codes", tensor)
        };

        using var results = GetSession().Run(inputs);

        // Extract waveform from first output tensor
        var outputTensor = results.First().AsTensor<float>();

        // Copy tensor values to a flat array
        var waveform = new float[outputTensor.Length];
        int i = 0;
        foreach (var sample in outputTensor)
            waveform[i++] = sample;

        return waveform;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
