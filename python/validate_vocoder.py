"""
Validate ONNX vocoder against PyTorch reference.

Compares PyTorch and ONNX Runtime outputs across multiple input shapes
to verify the exported model is numerically correct and handles dynamic
time dimensions properly.

Usage:
    python validate_vocoder.py
    python validate_vocoder.py --model-dir ./models/Qwen3-TTS-Tokenizer-12Hz
"""

import argparse
import sys

import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ONNX_PATH = Path(__file__).parent / "onnx_models" / "vocoder.onnx"

NUM_CODEBOOKS = 16
CODEBOOK_SIZE = 2048
UPSAMPLE_FACTOR = 1920


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(local_dir: str | None = None):
    """Load both the PyTorch decoder and the ONNX Runtime session."""
    from transformers import AutoModel
    import onnxruntime as ort

    if not ONNX_PATH.exists():
        print(f"✗ ONNX model not found at {ONNX_PATH}")
        print("  Run export_vocoder.py first.")
        sys.exit(1)

    repo = local_dir or TOKENIZER_REPO
    print(f"Loading PyTorch model from {repo} ...")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    decoder = model.decoder
    decoder.eval()

    print(f"Loading ONNX model from {ONNX_PATH} ...")
    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )

    return decoder, session


# ---------------------------------------------------------------------------
# Single comparison
# ---------------------------------------------------------------------------
def compare(decoder, session, batch_size: int, timesteps: int, label: str) -> dict:
    """Run inference on both backends and return comparison metrics."""
    codes = torch.randint(
        0, CODEBOOK_SIZE, (batch_size, NUM_CODEBOOKS, timesteps), dtype=torch.long
    )

    with torch.no_grad():
        wav_pt = decoder(codes).numpy()

    wav_onnx = session.run(["waveform"], {"codes": codes.numpy()})[0]

    max_err = float(np.max(np.abs(wav_pt - wav_onnx)))
    mean_err = float(np.mean(np.abs(wav_pt - wav_onnx)))
    expected_shape = (batch_size, 1, timesteps * UPSAMPLE_FACTOR)

    return {
        "label": label,
        "input_shape": (batch_size, NUM_CODEBOOKS, timesteps),
        "pt_shape": wav_pt.shape,
        "onnx_shape": wav_onnx.shape,
        "expected_shape": expected_shape,
        "shapes_ok": wav_pt.shape == wav_onnx.shape == expected_shape,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "pt_range": (float(wav_pt.min()), float(wav_pt.max())),
        "onnx_range": (float(wav_onnx.min()), float(wav_onnx.max())),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate vocoder ONNX export")
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Local directory with model weights (default: download from HF)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Vocoder — PyTorch vs ONNX Validation")
    print("=" * 60)

    decoder, session = load_models(args.model_dir)

    test_cases = [
        (1, 5, "Minimal (5 steps)"),
        (1, 10, "Short (10 steps)"),
        (1, 50, "Medium (50 steps)"),
        (1, 100, "Long (100 steps)"),
        (2, 10, "Batched (B=2, T=10)"),
    ]

    all_pass = True
    results = []

    for batch_size, timesteps, label in test_cases:
        print(f"\n--- {label} ---")
        print(f"  Input: codes ({batch_size}, {NUM_CODEBOOKS}, {timesteps})")

        r = compare(decoder, session, batch_size, timesteps, label)
        results.append(r)

        print(f"  PT shape:   {r['pt_shape']}")
        print(f"  ONNX shape: {r['onnx_shape']}")
        print(f"  Expected:   {r['expected_shape']}")
        print(f"  Shape OK:   {r['shapes_ok']}")
        print(f"  Max |err|:  {r['max_abs_error']:.6e}")
        print(f"  Mean |err|: {r['mean_abs_error']:.6e}")
        print(f"  PT range:   [{r['pt_range'][0]:.4f}, {r['pt_range'][1]:.4f}]")
        print(f"  ONNX range: [{r['onnx_range'][0]:.4f}, {r['onnx_range'][1]:.4f}]")

        if not r["shapes_ok"]:
            print("  ✗ FAIL: Shape mismatch")
            all_pass = False
        elif r["max_abs_error"] > 1e-3:
            print("  ✗ FAIL: Max error exceeds 1e-3")
            all_pass = False
        elif r["max_abs_error"] > 1e-4:
            print("  ~ WARN: Max error exceeds 1e-4 (likely float32 precision)")
        else:
            print("  ✓ PASS")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        ok = r["shapes_ok"] and r["max_abs_error"] < 1e-3
        status = "✓" if ok else "✗"
        print(f"  {status} {r['label']:30s}  max_err={r['max_abs_error']:.2e}")

    if all_pass:
        print("\n✓ All tests passed — ONNX model is numerically equivalent.")
    else:
        print("\n✗ Some tests failed. See details above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
