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
    /// Creates SessionOptions for CUDA GPU execution on the default device (GPU 0).
    /// Requires the <c>Microsoft.ML.OnnxRuntime.Gpu</c> NuGet package and CUDA Toolkit + cuDNN installed.
    /// Falls back to CPU if CUDA is unavailable.
    /// </summary>
    public static SessionOptions CreateCudaOptions() => CreateCudaOptions(0);

    /// <summary>
    /// Creates SessionOptions for CUDA GPU execution.
    /// Requires the <c>Microsoft.ML.OnnxRuntime.Gpu</c> NuGet package and CUDA Toolkit + cuDNN installed.
    /// Falls back to CPU if CUDA is unavailable.
    /// </summary>
    /// <param name="deviceId">GPU device ID.</param>
    public static SessionOptions CreateCudaOptions(int deviceId)
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
    /// Creates SessionOptions for DirectML GPU execution on the default device (GPU 0).
    /// Requires the <c>Microsoft.ML.OnnxRuntime.DirectML</c> NuGet package.
    /// Falls back to CPU if DirectML is unavailable.
    /// </summary>
    /// <remarks>
    /// Memory pattern optimization is disabled because the autoregressive language model uses
    /// dynamic KV-cache shapes that grow each decode step, which is incompatible with DML's
    /// memory pattern assumptions. Sequential execution mode is used for stability.
    /// </remarks>
    public static SessionOptions CreateDirectMlOptions() => CreateDirectMlOptions(0);

    /// <summary>
    /// Creates SessionOptions for DirectML GPU execution (Windows 10/11 — supports NVIDIA, AMD, and Intel GPUs).
    /// Requires the <c>Microsoft.ML.OnnxRuntime.DirectML</c> NuGet package.
    /// Falls back to CPU if DirectML is unavailable.
    /// </summary>
    /// <remarks>
    /// Memory pattern optimization is disabled because the autoregressive language model uses
    /// dynamic KV-cache shapes that grow each decode step, which is incompatible with DML's
    /// memory pattern assumptions. Sequential execution mode is used for stability.
    /// </remarks>
    /// <param name="deviceId">GPU device ID.</param>
    public static SessionOptions CreateDirectMlOptions(int deviceId)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableMemoryPattern = false,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
        options.AppendExecutionProvider_DML(deviceId);
        options.AppendExecutionProvider_CPU();
        return options;
    }
}
