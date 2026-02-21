# Prerequisites

## Required Software

| Requirement | Version | Install |
|-------------|---------|---------|
| **Python** | >= 3.10 | [python.org](https://www.python.org/downloads/) |
| **.NET SDK** | 10.0 | [dotnet.microsoft.com](https://dotnet.microsoft.com/download) |
| **Disk space** | ~6 GB | For ONNX models + NuGet packages |

> **No GPU required** — inference runs on CPU via ONNX Runtime. GPU acceleration is available by switching to `Microsoft.ML.OnnxRuntime.Gpu`.

## Verify Installation

```bash
python --version
dotnet --version
```

Both commands should return their respective version numbers. If either is missing, install from the links above and ensure they are on your system PATH.
