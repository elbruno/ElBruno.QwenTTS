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

## DirectML: Hybrid Mode

DirectML has limitations with certain ONNX operations in the vocoder model (dynamic padding). The ONNX models on HuggingFace are **pre-patched** for DirectML compatibility — no manual patching required.

However, the vocoder still requires CPU execution due to dynamic padding operations that DirectML cannot handle. Use the `vocoderSessionOptionsFactory` parameter to run the vocoder on CPU while keeping the language model on DirectML GPU:

```csharp
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
    vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);
```

Or with the options pattern:

```csharp
builder.Services.AddQwenTts(options =>
{
    options.ExecutionProvider = ExecutionProvider.DirectML;
});
// DirectML hybrid mode is applied automatically
```

### Advanced: Re-patching models manually

If you have custom or older ONNX models that aren't pre-patched, a Python script is included in the repository:

```bash
python python/patch_models_for_dml.py <model_directory>
```

The script applies: Reshape `-1` resolution, 1D ConvTranspose → 2D conversion, and causal mask optimization.

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
- `vocoderSessionOptionsFactory` — used for the vocoder session; defaults to `sessionOptionsFactory` when omitted **or explicitly set to `null`**. To force CPU for the vocoder, pass `OrtSessionHelper.CreateCpuOptions`.
- When both factories are `null` (default), sessions use CPU with full graph optimization
- The factory pattern lets you configure any execution provider ONNX Runtime supports
- A vocoder using a custom factory retries the specific runtime `Pad` / `Tensor shape.Size() must be >= 0` failure once on CPU, then reuses that CPU session. The language model stays on its configured provider. See [the Pad workaround below](#cuda-pad-node-failure--tensor-shapesize-must-be--0).

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
Delete your local model cache and re-download to get the pre-patched models. Also ensure you use `vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions` for the vocoder. If using older models, run `python python/patch_models_for_dml.py <model_dir>` to patch them manually.

### CUDA "Pad node" failure — `Tensor shape.Size() must be >= 0`

Synthesis may fail partway through generation on the CUDA execution provider with:

```
Non-zero status code returned while running Pad node. Name:'node_pad_1'
onnxruntime::Tensor::CalculateTensorStorageSize Tensor shape.Size() must be >= 0
```

The same text was reported to synthesize correctly on CPU. Inspection of the published 0.6B graphs
found `node_pad_1` in **`vocoder.onnx`**, with no Pad nodes in `talker_prefill.onnx` or
`talker_decode.onnx`. The original report used `vocoderSessionOptionsFactory: null`, which **inherits
the CUDA factory**; it did not establish that the failure occurs with a CPU vocoder. The precise
CUDA failure mechanism remains unconfirmed; disabling memory-pattern optimization is not a verified fix.

The library now handles this specific runtime failure when a custom vocoder session factory is in
use (including one inherited from `sessionOptionsFactory`):

- Retry only vocoder decoding on a new CPU session, using the already-generated audio codes.
- Reuse that CPU session for subsequent decodes; do not rerun or move the language model to CPU.
- Emit a `Trace` warning on the switch. Keep the original session alive until pipeline disposal so
  concurrent inference is not interrupted.
- Do not retry unrelated errors, session initialization failures, or cancelled requests. If CPU
  decoding also fails, propagate that error without another retry. Default CPU-only execution does
  not add a retry.

To avoid the failing CUDA vocoder attempt altogether (also useful with older library versions),
explicitly select hybrid execution:

```csharp
using var pipeline = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateCudaOptions,
    vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);
```

On Windows, [DirectML hybrid mode](#directml-hybrid-mode) is another option. CPU-only execution is
also available by leaving both factories unset.

The recovery path is regression-tested without GPU hardware; the original GTX 1650 Ti / CUDA 12.8
failure has not been reproduced on that hardware. Tracked in
[issue #73](https://github.com/elbruno/ElBruno.QwenTTS/issues/73).

### Mixed package errors
Only one ORT package can be referenced per project. Remove conflicting packages:
```bash
dotnet remove package Microsoft.ML.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```
