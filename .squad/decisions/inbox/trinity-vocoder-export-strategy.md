### Vocoder ONNX Export — Two-Attempt Strategy

**By:** Trinity (ML Engineer)
**Date:** 2026-02-21
**What:** The vocoder export script (`python/export_vocoder.py`) uses a two-pass strategy: standard trace-based `torch.onnx.export` at opset 17 first, falling back to `dynamo=True` at opset 18+ if tracing fails. Dynamic axes are set on both batch and time dimensions so the exported model handles variable-length audio.
**Why:** The decoder contains sliding-window attention and causal convolution padding that may produce data-dependent shapes during tracing. The dynamo backend handles these patterns more robustly. Having both paths lets us discover which works without manual iteration.
**Impact:** When running the export, if trace-based export works, we get maximum ONNX compatibility. If only dynamo works, the C# side may need ONNX Runtime ≥ 1.17 for full opset 18 support.
