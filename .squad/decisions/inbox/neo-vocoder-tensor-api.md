### 2026-02-21: Vocoder tensor API uses long[,,] not int[]
**By:** Neo
**What:** Changed `Vocoder.Decode()` signature from `int[]` to `long[,,]` (shape 1×16×T) to match the ONNX model's expected input tensor shape and dtype (int64). The pipeline reshapes the LM's flat output into this 3D array.
**Why:** The vocoder ONNX model expects `(B, 16, T_codes)` int64. Using the native shape avoids runtime reshaping inside the vocoder and keeps the contract explicit. The pipeline owns the reshape from the LM's flat output.
