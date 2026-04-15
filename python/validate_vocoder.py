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

from export_utils import configure_output_encoding

# ---------------------------------------------------------------------------
# Compatibility patches (same as export_vocoder.py)
# ---------------------------------------------------------------------------
import transformers.utils.generic as _tug
_orig_check = _tug.check_model_inputs

def _compat_check_model_inputs(func=None):
    if func is None:
        return _orig_check
    return _orig_check(func)

_tug.check_model_inputs = _compat_check_model_inputs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS
if "default" not in _ROPE_FNS:
    def _compute_default_rope(config=None, device=None, **kwargs):
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
        return inv_freq, 1.0
    _ROPE_FNS["default"] = _compute_default_rope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
ONNX_PATH = Path(__file__).parent / "onnx_models" / "vocoder.onnx"

NUM_CODEBOOKS = 16
CODEBOOK_SIZE = 2048
UPSAMPLE_FACTOR = 1920


# ---------------------------------------------------------------------------
# VocoderOnnxWrapper (must match the wrapper used during export)
# ---------------------------------------------------------------------------
class VocoderOnnxWrapper(torch.nn.Module):
    """ONNX-friendly wrapper that bypasses create_causal_mask for dynamic shapes.

    Must be identical to the wrapper in export_vocoder.py so that PyTorch
    reference output matches the ONNX graph structure.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        config = decoder.pre_transformer.config
        self.sliding_window = getattr(config, "sliding_window", None) or 9999

    def _make_sliding_window_mask(self, seq_len: int, device: torch.device):
        pos = torch.arange(seq_len, device=device)
        q_pos = pos.unsqueeze(1)
        kv_pos = pos.unsqueeze(0)
        diff = q_pos - kv_pos
        mask = (diff >= 0) & (diff < self.sliding_window)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, codes):
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)

        transformer = self.decoder.pre_transformer
        inputs_embeds = transformer.input_proj(hidden)

        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = transformer.rotary_emb(inputs_embeds, position_ids)

        attn_mask = self._make_sliding_window_mask(seq_len, device)

        hidden_states = inputs_embeds
        for layer in transformer.layers[: transformer.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        hidden_states = transformer.norm(hidden_states)
        hidden_states = transformer.output_proj(hidden_states)

        hidden = hidden_states.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


# ---------------------------------------------------------------------------
# RoPE inv_freq fix
# ---------------------------------------------------------------------------
def _fix_rope_inv_freq(decoder):
    """Recompute correct RoPE inv_freq (non-persistent buffer gets garbage
    after meta-device loading in from_pretrained)."""
    rotary = decoder.pre_transformer.rotary_emb
    config = decoder.pre_transformer.config
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    theta = config.rope_theta
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    rotary.inv_freq = inv_freq
    print(f"  Fixed RoPE inv_freq: shape={inv_freq.shape}, "
          f"range=[{inv_freq.min():.6f}, {inv_freq.max():.6f}]")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(local_dir: str | None = None):
    """Load both the PyTorch decoder (wrapped) and the ONNX Runtime session."""
    from qwen_tts.core import Qwen3TTSTokenizerV2Model, Qwen3TTSTokenizerV2Config
    import onnxruntime as ort

    if not ONNX_PATH.exists():
        print(f"✗ ONNX model not found at {ONNX_PATH}")
        print("  Run export_vocoder.py first.")
        sys.exit(1)

    repo = local_dir or TOKENIZER_REPO
    print(f"Loading PyTorch model from {repo} ...")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(repo)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(repo, config=config)
    decoder = model.decoder
    decoder.eval()

    # Fix non-persistent RoPE inv_freq (garbage after meta-device loading)
    _fix_rope_inv_freq(decoder)

    # Use the same wrapper that was used during ONNX export
    wrapper = VocoderOnnxWrapper(decoder)
    wrapper.eval()

    print(f"Loading ONNX model from {ONNX_PATH} ...")
    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )

    return wrapper, session


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
    configure_output_encoding()
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
