using Microsoft.ML.OnnxRuntime;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Helper methods for creating ONNX Runtime SessionOptions with different execution providers.
/// 
/// Usage: Pass the desired factory method to TtsPipeline or VoiceClonePipeline constructors.
/// The consumer must reference the appropriate NuGet package for their chosen GPU backend:
///   - CPU (default): Microsoft.ML.OnnxRuntime
///   - CUDA: Microsoft.ML.OnnxRuntime.Gpu
///   - DirectML: Microsoft.ML.OnnxRuntime.DirectML
/// 
/// Note: GPU NuGet packages are mutually exclusive — only one can be referenced per project.
/// </summary>
public static class OrtSessionHelper
{
    /// <summary>
    /// Creates SessionOptions for CPU execution (default behavior).
    /// </summary>
    public static SessionOptions CreateCpuOptions()
    {
        return new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
    }

    /// <summary>
    /// Creates SessionOptions for CUDA GPU execution.
    /// Requires the <c>Microsoft.ML.OnnxRuntime.Gpu</c> NuGet package and CUDA Toolkit + cuDNN installed.
    /// Falls back to CPU if CUDA is unavailable.
    /// </summary>
    /// <param name="deviceId">GPU device ID (default: 0).</param>
    public static SessionOptions CreateCudaOptions(int deviceId = 0)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        options.AppendExecutionProvider_CUDA(deviceId);
        options.AppendExecutionProvider_CPU();
        return options;
    }

    /// <summary>
    /// Creates SessionOptions for DirectML GPU execution (Windows 10/11 — supports NVIDIA, AMD, and Intel GPUs).
    /// Requires the <c>Microsoft.ML.OnnxRuntime.DirectML</c> NuGet package.
    /// Falls back to CPU if DirectML is unavailable.
    /// </summary>
    /// <param name="deviceId">GPU device ID (default: 0).</param>
    public static SessionOptions CreateDirectMlOptions(int deviceId = 0)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        options.AppendExecutionProvider_DML(deviceId);
        options.AppendExecutionProvider_CPU();
        return options;
    }
}
