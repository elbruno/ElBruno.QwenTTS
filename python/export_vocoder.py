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

from export_utils import configure_output_encoding

# Compatibility patch: transformers 5.5+ changed check_model_inputs from a
# decorator factory (@check_model_inputs()) to a plain decorator (@check_model_inputs).
# The qwen_tts package still uses the old factory style, so we monkey-patch it
# to accept both calling conventions before qwen_tts is imported.
import transformers.utils.generic as _tug
_orig_check = _tug.check_model_inputs

def _compat_check_model_inputs(func=None):
    if func is None:
        return lambda fn: fn  # called as @check_model_inputs() → identity decorator
    return func  # called as @check_model_inputs → return function unchanged

_tug.check_model_inputs = _compat_check_model_inputs

# Compatibility patch: transformers 5.5+ removed "default" from ROPE_INIT_FUNCTIONS.
# The qwen_tts package uses "default" when no rope_scaling is configured. The default
# ROPE is just standard inv_freq = 1/(theta^(2i/dim)) with no scaling.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS
if "default" not in _ROPE_FNS:
    def _compute_default_rope(config=None, device=None, **kwargs):
        import math
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
        return inv_freq, 1.0
    _ROPE_FNS["default"] = _compute_default_rope

# Compatibility patch: sdpa_mask BC path crashes during ONNX tracing when q_length
# is a 0-d tensor (scalar). The BC check does q_length.shape[0] which fails on
# scalars. Wrap sdpa_mask to convert scalar tensors to ints before calling.
import transformers.masking_utils as _masking
_orig_sdpa_mask = _masking.sdpa_mask

def _patched_sdpa_mask(q_length=None, **kwargs):
    if isinstance(q_length, torch.Tensor):
        if q_length.dim() == 0:
            q_length = int(q_length.item())
        else:
            q_length = q_length.shape[0]
            kwargs.setdefault("q_offset", int(q_length[0]) if len(q_length) > 0 else 0)
    return _orig_sdpa_mask(q_length=q_length, **kwargs)

_masking.sdpa_mask = _patched_sdpa_mask
# Also patch in ALL_MASK_ATTENTION_FUNCTIONS registry if sdpa is registered
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS as _MASK_FNS
if 'sdpa' in _MASK_FNS:
    _MASK_FNS['sdpa'] = _patched_sdpa_mask

# Compatibility patch: torch.diff is not supported in ONNX trace export.
# Replace it with equivalent cat+subtract ops. Only used in masking_utils
# for packed-sequence detection via position_ids diff.
_orig_torch_diff = torch.diff

def _onnx_safe_diff(input, n=1, dim=-1, prepend=None, append=None):
    if n != 1:
        return _orig_torch_diff(input, n=n, dim=dim, prepend=prepend, append=append)
    parts = []
    if prepend is not None:
        parts.append(prepend)
    parts.append(input)
    if append is not None:
        parts.append(append)
    combined = torch.cat(parts, dim=dim) if len(parts) > 1 else input
    return torch.narrow(combined, dim, 1, combined.size(dim) - 1) - torch.narrow(combined, dim, 0, combined.size(dim) - 1)

torch.diff = _onnx_safe_diff

# Compatibility patch: ONNX CumSum doesn't accept bool tensors. PyTorch allows
# cumsum on bools (treating as 0/1), but ONNX requires numeric types. Wrap
# Tensor.cumsum to cast bool→int64 before the operation.
_orig_cumsum = torch.Tensor.cumsum

def _onnx_safe_cumsum(self, dim, dtype=None):
    if self.dtype == torch.bool:
        return _orig_cumsum(self.to(torch.int64), dim, dtype=dtype)
    return _orig_cumsum(self, dim, dtype=dtype)

torch.Tensor.cumsum = _onnx_safe_cumsum

# Compatibility patch: ONNX trace doesn't support scaled_dot_product_attention with
# enable_gqa=True. The GQA path is triggered when attention_mask is None on torch>=2.5.
# Disable it during export — the model uses MHA (num_kv_heads == num_heads) anyway.
import transformers.integrations.sdpa_attention as _sdpa_mod
_sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False

# Register vmap-free masking for ONNX export compatibility with transformers 4.57+
# In transformers 5.5+, sdpa_mask already defaults to use_vmap=False, so the
# executorch workaround is unnecessary. We try it for older versions and skip if absent.
try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.executorch import sdpa_mask_without_vmap
    ALL_MASK_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', ALL_ATTENTION_FUNCTIONS['sdpa'])
    _VMAP_WORKAROUND = 'sdpa_without_vmap'
except ImportError:
    _VMAP_WORKAROUND = None  # Not needed in transformers 5.5+ (use_vmap=False is default)

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
class VocoderOnnxWrapper(torch.nn.Module):
    """ONNX-friendly wrapper that bypasses create_causal_mask to enable dynamic shapes.

    The standard transformer forward creates attention masks and position embeddings
    using Python-level shape arithmetic (torch.arange, math.ceil, etc.) that gets
    baked as constants during torch.onnx.export tracing. This wrapper manually runs
    the transformer layers with:
    - position_ids derived from input via torch.arange (ONNX-traceable Shape→Range)
    - Sliding window causal mask built from tensor ops (ONNX-traceable)
    - No cache (single-pass vocoder inference)
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        config = decoder.pre_transformer.config
        self.sliding_window = getattr(config, "sliding_window", None) or 9999

    def _make_sliding_window_mask(self, seq_len: int, device: torch.device):
        """Build a sliding-window causal boolean mask: (1, 1, S, S).

        attend[q, kv] = True iff (kv <= q) AND (q - kv < window_size)
        """
        pos = torch.arange(seq_len, device=device)
        q_pos = pos.unsqueeze(1)   # (S, 1)
        kv_pos = pos.unsqueeze(0)  # (1, S)
        diff = q_pos - kv_pos
        mask = (diff >= 0) & (diff < self.sliding_window)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def forward(self, codes):
        # --- Quantizer + pre-conv (same as original) ---
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)

        # --- Transformer (bypassing create_causal_mask) ---
        transformer = self.decoder.pre_transformer
        inputs_embeds = transformer.input_proj(hidden)

        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = transformer.rotary_emb(inputs_embeds, position_ids)

        # Build sliding-window causal mask (all layers use sliding_attention)
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

        # --- Upsample + BigVGAN decoder (same as original) ---
        hidden = hidden_states.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def _fix_rope_inv_freq(decoder):
    """Recompute and set correct RoPE inv_freq on the decoder's rotary embedding.

    When transformers loads the model via from_pretrained with device=meta,
    non-persistent buffers (like inv_freq) become uninitialized meta tensors.
    Since inv_freq is registered as persistent=False, it is NOT in the state dict
    and never gets overwritten with real values. This leaves it with garbage data
    that differs across processes. We must explicitly compute the correct values.
    """
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


def load_decoder(local_dir: str | None = None):
    """Load the Qwen3-TTS-Tokenizer-12Hz model and extract the decoder."""
    from qwen_tts.core import Qwen3TTSTokenizerV2Model, Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalConvNet,
    )

    # Patch CausalConvNet to avoid Python math.ceil in _get_extra_padding_for_conv1d.
    # All instances in this model use stride=1 so extra_padding is always 0.
    # Using Python math causes the tracer to bake shapes as constants.
    Qwen3TTSTokenizerV2CausalConvNet._get_extra_padding_for_conv1d = lambda self, x: 0

    repo = local_dir or TOKENIZER_REPO
    print(f"Loading model from {repo} ...")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(repo)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(repo, config=config)
    decoder = model.decoder
    decoder.eval()

    # Fix non-persistent RoPE inv_freq (garbage after meta-device loading)
    _fix_rope_inv_freq(decoder)
    # Patch transformer layers for vmap-free masking (ONNX trace compatibility)
    if _VMAP_WORKAROUND and hasattr(decoder, 'pre_transformer'):
        print(f"  Patching attention to '{_VMAP_WORKAROUND}' for ONNX trace ...")
        decoder.pre_transformer.config._attn_implementation = _VMAP_WORKAROUND
        if hasattr(decoder.pre_transformer, 'layers'):
            for layer in decoder.pre_transformer.layers:
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.config._attn_implementation = _VMAP_WORKAROUND
    elif not _VMAP_WORKAROUND:
        print("  vmap workaround not needed (transformers 5.5+ detected)")
    print(f"  Decoder class: {type(decoder).__name__}")
    print(f"  Total upsample factor: {decoder.total_upsample}")

    # Wrap in ONNX-friendly module
    wrapper = VocoderOnnxWrapper(decoder)
    wrapper.eval()

    # Verify wrapper produces same output as original
    test_codes = torch.randint(0, CODEBOOK_SIZE, (1, NUM_CODEBOOKS, 10), dtype=torch.long)
    with torch.no_grad():
        orig_out = decoder(test_codes)
        wrap_out = wrapper(test_codes)
    max_diff = (orig_out - wrap_out).abs().max().item()
    print(f"  Wrapper vs original max diff: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Wrapper output differs from original by {max_diff}"

    return wrapper


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
    upsample = getattr(decoder, 'total_upsample', None) or getattr(getattr(decoder, 'decoder', None), 'total_upsample', 1920)
    print(f"Upsample check: {codes.shape[-1]} × {upsample} = "
          f"{codes.shape[-1] * upsample}  (got {wav_pt.shape[-1]})")

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
            do_constant_folding=False,
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
        "--output-dir", type=str, default=None,
        help="Directory to save vocoder.onnx (default: onnx_models/)",
    )
    p.add_argument(
        "--skip-validate", action="store_true",
        help="Skip ONNX Runtime validation after export",
    )
    return p.parse_args()


def main():
    configure_output_encoding()
    global ONNX_OUTPUT_DIR, ONNX_OUTPUT_PATH
    args = parse_args()

    if args.output_dir:
        ONNX_OUTPUT_DIR = Path(args.output_dir)
        ONNX_OUTPUT_PATH = ONNX_OUTPUT_DIR / "vocoder.onnx"

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
