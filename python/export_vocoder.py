"""
Export Qwen3-TTS Tokenizer-12Hz vocoder decoder to ONNX.

Input:  codes — shape (1, 16, T) int64, values in [0, 2047]
Output: waveform — shape (1, 1, T*1920) float32, values in [-1, 1]

The vocoder (Qwen3TTSTokenizerV2Decoder) is a single forward pass:
  codes → RVQ dequantize → Conv1d → 8-layer Transformer
        → TransConv upsample (2×2) → BigVGAN-style decoder (8×5×4×3)
        → waveform (1920× total upsample, 12 Hz codes → 24 kHz PCM)

Usage:
    python export_vocoder.py
    python export_vocoder.py --timesteps 20 --opset 18
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ONNX_OUTPUT_DIR = Path(__file__).parent / "onnx_models"
ONNX_OUTPUT_PATH = ONNX_OUTPUT_DIR / "vocoder.onnx"

NUM_CODEBOOKS = 16
CODEBOOK_SIZE = 2048
DEFAULT_TIMESTEPS = 10


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_decoder(local_dir: str | None = None):
    """Load the Qwen3-TTS-Tokenizer-12Hz model and extract the decoder."""
    from transformers import AutoModel

    repo = local_dir or TOKENIZER_REPO
    print(f"Loading model from {repo} ...")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    decoder = model.decoder
    decoder.eval()
    print(f"  Decoder class: {type(decoder).__name__}")
    print(f"  Total upsample factor: {decoder.total_upsample}")
    return decoder


# ---------------------------------------------------------------------------
# Dummy input
# ---------------------------------------------------------------------------
def create_dummy_input(batch_size: int = 1, timesteps: int = DEFAULT_TIMESTEPS):
    """Create dummy RVQ codes for the vocoder."""
    return torch.randint(
        0, CODEBOOK_SIZE, (batch_size, NUM_CODEBOOKS, timesteps), dtype=torch.long
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(decoder, codes, opset: int = 17) -> bool:
    """Export vocoder decoder to ONNX with fallback strategies."""
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInput shape:  {tuple(codes.shape)}  (batch, codebooks, timesteps)")
    with torch.no_grad():
        wav_pt = decoder(codes)
    print(f"Output shape: {tuple(wav_pt.shape)}  (batch, channels, samples)")
    print(f"Upsample check: {codes.shape[-1]} × {decoder.total_upsample} = "
          f"{codes.shape[-1] * decoder.total_upsample}  (got {wav_pt.shape[-1]})")

    dynamic_axes = {
        "codes":    {0: "batch_size", 2: "num_timesteps"},
        "waveform": {0: "batch_size", 2: "num_samples"},
    }

    # --- Attempt 1: standard trace-based export ---
    try:
        print(f"\n[1/2] torch.onnx.export (trace, opset {opset}) ...")
        t0 = time.time()
        torch.onnx.export(
            decoder,
            (codes,),
            str(ONNX_OUTPUT_PATH),
            opset_version=opset,
            input_names=["codes"],
            output_names=["waveform"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        elapsed = time.time() - t0
        size_mb = ONNX_OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported in {elapsed:.1f}s  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Trace export failed: {e}")

    # --- Attempt 2: dynamo-based export (opset 18+) ---
    try:
        dynamo_opset = max(opset, 18)
        print(f"\n[2/2] torch.onnx.export (dynamo, opset {dynamo_opset}) ...")
        t0 = time.time()
        torch.onnx.export(
            decoder,
            (codes,),
            str(ONNX_OUTPUT_PATH),
            opset_version=dynamo_opset,
            input_names=["codes"],
            output_names=["waveform"],
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )
        elapsed = time.time() - t0
        size_mb = ONNX_OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✓ Exported (dynamo) in {elapsed:.1f}s  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Dynamo export also failed: {e}")

    print("\n--- Diagnosis ---")
    print("The decoder may contain ops unsupported by ONNX tracing.")
    print("Likely culprits:")
    print("  - Sliding-window attention mask (data-dependent shapes)")
    print("  - Causal convolution padding (_get_extra_padding_for_conv1d)")
    print("  - SnakeBeta activation (should be fine — sin/pow are standard)")
    print("Consider exporting sub-components separately or wrapping")
    print("the problematic layers with torch.jit.script annotations.")
    return False


# ---------------------------------------------------------------------------
# Quick validation (full validation in validate_vocoder.py)
# ---------------------------------------------------------------------------
def quick_validate(decoder, codes) -> bool:
    """Quick numerical check: PyTorch vs ONNX Runtime."""
    import onnxruntime as ort

    if not ONNX_OUTPUT_PATH.exists():
        print("\nSkipping validation — no ONNX file found.")
        return False

    print("\nQuick validation ...")

    with torch.no_grad():
        wav_pt = decoder(codes).numpy()

    session = ort.InferenceSession(
        str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"]
    )
    wav_onnx = session.run(["waveform"], {"codes": codes.numpy()})[0]

    max_err = float(np.max(np.abs(wav_pt - wav_onnx)))
    mean_err = float(np.mean(np.abs(wav_pt - wav_onnx)))

    print(f"  PT shape:       {wav_pt.shape}")
    print(f"  ONNX shape:     {wav_onnx.shape}")
    print(f"  Max |error|:    {max_err:.6e}")
    print(f"  Mean |error|:   {mean_err:.6e}")

    ok = wav_pt.shape == wav_onnx.shape and max_err < 1e-3
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Export Qwen3-TTS vocoder to ONNX")
    p.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help="Number of code timesteps for dummy input (default: 10)",
    )
    p.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    p.add_argument(
        "--model-dir", type=str, default=None,
        help="Local directory with model weights (default: download from HF)",
    )
    p.add_argument(
        "--skip-validate", action="store_true",
        help="Skip ONNX Runtime validation after export",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Qwen3-TTS Vocoder Decoder → ONNX Export")
    print("=" * 60)

    decoder = load_decoder(args.model_dir)
    codes = create_dummy_input(timesteps=args.timesteps)

    with torch.no_grad():
        success = export_onnx(decoder, codes, opset=args.opset)

    if success and not args.skip_validate:
        quick_validate(decoder, codes)

    print("\nDone." if success else "\nExport failed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
