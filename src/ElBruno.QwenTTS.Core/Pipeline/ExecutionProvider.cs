namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// GPU/CPU execution provider for ONNX Runtime sessions.
/// </summary>
public enum ExecutionProvider
{
    /// <summary>CPU execution (default). No additional NuGet packages required.</summary>
    Cpu,

    /// <summary>
    /// NVIDIA GPU via CUDA. Requires <c>Microsoft.ML.OnnxRuntime.Gpu</c> NuGet package
    /// and CUDA Toolkit + cuDNN installed.
    /// </summary>
    Cuda,

    /// <summary>
    /// DirectML GPU (NVIDIA, AMD, Intel on Windows 10/11).
    /// Requires <c>Microsoft.ML.OnnxRuntime.DirectML</c> NuGet package.
    /// Models must be patched with <c>python/patch_models_for_dml.py</c>.
    /// Uses hybrid mode: GPU for language model, CPU for vocoder.
    /// </summary>
    DirectML
}
