# GPU Acceleration

ElBruno.QwenTTS supports GPU acceleration through ONNX Runtime execution providers. The same ONNX models work on both CPU and GPU — no re-export needed.

## Quick Start

### 1. Choose your GPU backend

| Package | Backend | Supported GPUs | OS | Prerequisites |
|---------|---------|---------------|-----|--------------|
| `Microsoft.ML.OnnxRuntime` | CPU | — | All | None |
| `Microsoft.ML.OnnxRuntime.Gpu` | CUDA | NVIDIA | Win/Linux | CUDA Toolkit + cuDNN |
| `Microsoft.ML.OnnxRuntime.DirectML` | DirectML | NVIDIA, AMD, Intel | Windows 10/11 | None |

> **Important:** These packages are **mutually exclusive** — only reference **one** in your project.

### 2. Swap the NuGet package

Replace the CPU package with your GPU backend:

```xml
<!-- In your .csproj — choose ONE: -->

<!-- CPU (default) -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.24.2" />

<!-- NVIDIA GPU via CUDA -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.24.2" />

<!-- Any GPU via DirectML (Windows only) -->
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.24.2" />
```

### 3. Configure SessionOptions

Pass a `sessionOptionsFactory` to the pipeline constructor or factory:

```csharp
using ElBruno.QwenTTS.Pipeline;
using Microsoft.ML.OnnxRuntime;

// CPU (default — no changes needed)
var tts = await TtsPipeline.CreateAsync();

// CUDA GPU
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateCudaOptions);

// DirectML GPU (Windows) — hybrid mode: GPU for LM, CPU for vocoder
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
    vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);

// Custom configuration
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: () =>
    {
        var opts = new SessionOptions();
        opts.AppendExecutionProvider_CUDA(deviceId: 1); // second GPU
        opts.AppendExecutionProvider_CPU(); // CPU fallback
        return opts;
    });
```

### Voice Cloning with GPU

The same pattern works for `VoiceClonePipeline`:

```csharp
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

var cloner = await VoiceClonePipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateCudaOptions);
```

## DirectML: Model Patching Required

DirectML has limitations with certain ONNX operations:
- **Reshape with `-1`** (inferred dimensions) combined with dynamic Shape ops
- **1D ConvTranspose** in the vocoder model

A Python patch script resolves these issues:

```bash
# Patch models for DirectML compatibility
python python/patch_models_for_dml.py <model_directory>

# Or patch to a separate output directory
python python/patch_models_for_dml.py <model_directory> --output <output_directory>
```

The script applies these transformations:
1. **Resolves `-1` in Reshape shapes** — traces MatMul weights to compute Q/K/V head counts and merge dimensions
2. **Converts 1D ConvTranspose to 2D** — wraps with Unsqueeze/Squeeze for DML compatibility
3. **Replaces `Reshape([-1,1])` with `Unsqueeze`** — optimizes causal mask construction

> **Note:** Even with patched models, the vocoder requires CPU execution due to dynamic padding operations. Use the `vocoderSessionOptionsFactory` parameter to run the vocoder on CPU while keeping the language model on DirectML GPU:
>
> ```csharp
> var tts = new TtsPipeline(modelDir,
>     sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
>     vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);
> ```

## Helper Methods

`OrtSessionHelper` provides convenience methods for common configurations:

| Method | Description |
|--------|-------------|
| `CreateCpuOptions()` | CPU with max graph optimization (default) |
| `CreateCudaOptions(deviceId)` | CUDA provider + CPU fallback |
| `CreateDirectMlOptions(deviceId)` | DirectML provider + CPU fallback |

All methods set `GraphOptimizationLevel.ORT_ENABLE_ALL` and include CPU as a fallback provider.

## How It Works

- `TtsPipeline` and `VoiceClonePipeline` accept optional `Func<SessionOptions>?` parameters
- `sessionOptionsFactory` — used for language model sessions (prefill, decode, code predictor)
- `vocoderSessionOptionsFactory` — used for the vocoder session; defaults to `sessionOptionsFactory` if not specified
- When `null` (default), sessions use CPU with full graph optimization
- The factory pattern lets you configure any execution provider ONNX Runtime supports

## Performance Notes

- **CUDA** offers the best performance for NVIDIA GPUs but requires driver/toolkit installation
- **DirectML** works out-of-the-box on Windows with any GPU vendor (requires model patching + CPU vocoder fallback)
- The language model (3 sessions: prefill, decode, code_predictor) benefits most from GPU acceleration
- The vocoder is relatively lightweight and may not see as much speedup
- First inference is slower due to GPU kernel compilation; subsequent calls are faster

## Troubleshooting

### "CUDA execution provider is not enabled"
Install `Microsoft.ML.OnnxRuntime.Gpu` instead of `Microsoft.ML.OnnxRuntime`, and ensure CUDA Toolkit + cuDNN are installed.

### "DML execution provider is not enabled"
Install `Microsoft.ML.OnnxRuntime.DirectML` instead of `Microsoft.ML.OnnxRuntime`.

### DirectML Reshape crash
Run `python python/patch_models_for_dml.py <model_dir>` to patch models, and use `vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions` for the vocoder.

### Mixed package errors
Only one ORT package can be referenced per project. Remove conflicting packages:
```bash
dotnet remove package Microsoft.ML.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```
