# Troubleshooting

## Common Issues

| Problem | Solution |
|---------|----------|
| `python: command not found` | Install Python 3.10+ and add to PATH |
| `dotnet: command not found` | Install [.NET 10 SDK](https://dotnet.microsoft.com/download) |
| Download hangs or fails | Re-run `python download_onnx_models.py` -- it skips already-downloaded files |
| `huggingface_hub` import error | Run `pip install huggingface_hub` |
| Out of memory during inference | Close other applications -- the LM loads ~3.4 GB into RAM |
| Wrong speaker name | Use lowercase: `ryan`, `serena`, etc. (see [CLI Reference](cli-reference.md)) |
| `invalid expand shape` in C# ORT | Re-export LM models with `python/reexport_lm_novmap.py` (vmap-free masking) |
| `Not a valid NPY file (bad magic)` | Ensure NpyReader uses byte array `[0x93, ...]` not UTF-8 string literal |

## ONNX Runtime C# Expand Bug

If you see `invalid expand shape` errors when running the C# app, the ONNX models were exported with `torch.vmap`-generated Expand nodes that are incompatible with ORT's C# binding.

**Fix:** Re-export the LM models using `python/reexport_lm_novmap.py` which uses vmap-free masking. See [Exporting Models](exporting-models.md) for details.

## NPY Magic Byte Bug

If you see `Not a valid NPY file (bad magic)`, the C# NpyReader may be using a UTF-8 string literal `"\x93NUMPY"u8` which encodes the byte 0x93 as two UTF-8 bytes (0xC2, 0x93). The fix is to use an explicit byte array:

```csharp
ReadOnlySpan<byte> expected = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
```

## Model Download Issues

- The setup script and download script both support resuming interrupted downloads
- Files are verified by checking existence and file size
- If a file appears corrupted, delete it and re-run the download script
- The total download size is approximately 5.5 GB
