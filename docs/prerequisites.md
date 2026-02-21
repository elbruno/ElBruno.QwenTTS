# Prerequisites

## Required Software

| Requirement | Version | Install |
|-------------|---------|---------|
| **.NET SDK** | 10.0 | [dotnet.microsoft.com](https://dotnet.microsoft.com/download) |
| **Disk space** | ~6 GB | For ONNX models + NuGet packages |
| **Python** *(optional)* | >= 3.10 | [python.org](https://www.python.org/downloads/) — only needed for the alternative Python setup or ONNX re-export |

> **No GPU required** — inference runs on CPU via ONNX Runtime. GPU acceleration is available by switching to `Microsoft.ML.OnnxRuntime.Gpu`.
>
> **No Python required** — the C# library downloads models automatically from HuggingFace. Python is only needed if you want to use the alternative `setup_environment.py` script or re-export models.

## Verify Installation

```bash
dotnet --version
```

The .NET SDK 10.0 or later must be installed. Models are downloaded automatically by the library on first run.
