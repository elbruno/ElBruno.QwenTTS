namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Configuration options for QwenTTS pipeline.
/// </summary>
public class QwenTtsOptions
{
    /// <summary>
    /// Path to local model directory. When null, uses the default shared location
    /// (<c>%LOCALAPPDATA%/ElBruno.QwenTTS/models</c> on Windows, <c>~/.local/share/ElBruno.QwenTTS/models</c> on Linux/macOS).
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// HuggingFace repository ID for model download.
    /// Default: <c>elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX</c>.
    /// </summary>
    public string HuggingFaceRepo { get; set; } = ModelDownloader.DefaultRepoId;

    /// <summary>
    /// Execution provider for GPU/CPU selection. Default: <see cref="Pipeline.ExecutionProvider.Cpu"/>.
    /// </summary>
    public ExecutionProvider ExecutionProvider { get; set; } = ExecutionProvider.Cpu;

    /// <summary>
    /// GPU device ID when using CUDA or DirectML. Default: 0.
    /// </summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>
    /// Optional custom session options factory. When set, overrides <see cref="ExecutionProvider"/>.
    /// Use this for advanced scenarios not covered by the enum.
    /// </summary>
    public Func<Microsoft.ML.OnnxRuntime.SessionOptions>? SessionOptionsFactory { get; set; }
}
