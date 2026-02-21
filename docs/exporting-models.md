# Exporting Models

## Using Pre-Exported Models

The easiest way is to download the pre-exported ONNX models from HuggingFace:

```bash
python setup_environment.py
```

Or manually:

```bash
cd python
pip install huggingface_hub
python download_onnx_models.py
```

## Re-Exporting from PyTorch Weights

If you need to re-export the ONNX models from PyTorch weights (e.g., for a different model variant):

```bash
cd python
pip install -r requirements.txt
python download_models.py
python reexport_lm_novmap.py
python export_vocoder.py
python export_embeddings.py
python extract_tokenizer.py
```

### Important: vmap-free Masking

The LM models **must** be exported with `reexport_lm_novmap.py` (not `export_lm.py`) to work with C# ONNX Runtime.

The standard `export_lm.py` produces models that use `torch.vmap`-generated Expand nodes which are incompatible with ORT's C# binding. The `reexport_lm_novmap.py` script uses the ExecuTorch `sdpa_mask_without_vmap` approach to produce clean ONNX graphs compatible with both Python and C# ORT.

### Export Scripts

| Script | Output | Notes |
|--------|--------|-------|
| `reexport_lm_novmap.py` | `talker_prefill.onnx`, `talker_decode.onnx`, `code_predictor.onnx` | **Use this** for C# compatible models |
| `export_lm.py` | Same files | Original export, Python ORT only |
| `export_vocoder.py` | `vocoder.onnx` | Works in both Python and C# ORT |
| `export_embeddings.py` | `embeddings/*.npy` | Embedding weight extraction |
| `extract_tokenizer.py` | `tokenizer/vocab.json`, `tokenizer/merges.txt` | BPE tokenizer artifacts |

## Uploading to HuggingFace

To upload your exported models to HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli login
python upload_to_hf.py --repo-id your-username/your-repo-name
```

You need a write token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
