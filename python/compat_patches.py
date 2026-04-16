"""
Shared compatibility patches for ONNX export scripts.

These patches resolve incompatibilities between the qwen_tts package and newer
versions of transformers (4.57+, 5.5+) during torch.onnx.export tracing.

Import this module BEFORE importing qwen_tts or any model code:

    import compat_patches  # noqa: F401 — patches applied on import

Patches applied:
    1. check_model_inputs: transformers 5.5+ changed from decorator factory
       (@check_model_inputs()) to plain decorator (@check_model_inputs).
    2. ROPE_INIT_FUNCTIONS["default"]: removed in transformers 5.5+, but
       qwen_tts still references it.
    3. sdpa_mask: ONNX tracing creates 0-d tensors for q_length, which crashes
       the BC check (q_length.shape[0] on a scalar).
    4. torch.diff: not supported in ONNX trace export; replaced with
       equivalent cat+subtract.
    5. torch.Tensor.cumsum: ONNX CumSum doesn't accept bool tensors.
    6. use_gqa_in_sdpa: disable GQA in SDPA (not needed, and breaks tracing
       when attention_mask is None on torch>=2.5).
    7. vmap-free masking: register sdpa_without_vmap for ONNX trace
       compatibility. The vmap-based masking in transformers 4.57+
       crashes during torch.onnx.export with:
       RuntimeError: invalid unordered_map<K, T> key
    8. torch.cdist: ONNX symbolic export can't determine row_size_x1
       statically, causing AssertionError in cdist. Replaced with
       manual pairwise distance using basic tensor ops.
"""

import torch

# ═══════════════════════════════════════════════════════════════════════════
# 1. check_model_inputs compat
# ═══════════════════════════════════════════════════════════════════════════
import transformers.utils.generic as _tug

_orig_check = _tug.check_model_inputs


def _compat_check_model_inputs(func=None):
    if func is None:
        return _orig_check  # called as @check_model_inputs() → return decorator
    return _orig_check(func)  # called as @check_model_inputs → apply directly


_tug.check_model_inputs = _compat_check_model_inputs


# ═══════════════════════════════════════════════════════════════════════════
# 2. ROPE_INIT_FUNCTIONS["default"]
# ═══════════════════════════════════════════════════════════════════════════
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS

if "default" not in _ROPE_FNS:
    def _compute_default_rope(config=None, device=None, **kwargs):
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim)
        )
        return inv_freq, 1.0

    _ROPE_FNS["default"] = _compute_default_rope


# ═══════════════════════════════════════════════════════════════════════════
# 3. sdpa_mask scalar q_length fix
# ═══════════════════════════════════════════════════════════════════════════
import transformers.masking_utils as _masking

_orig_sdpa_mask = _masking.sdpa_mask


def _patched_sdpa_mask(q_length=None, **kwargs):
    if isinstance(q_length, torch.Tensor):
        if q_length.dim() == 0:
            q_length = int(q_length.item())
        else:
            q_length = q_length.shape[0]
    return _orig_sdpa_mask(q_length=q_length, **kwargs)


_masking.sdpa_mask = _patched_sdpa_mask
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS as _MASK_FNS

if "sdpa" in _MASK_FNS:
    _MASK_FNS["sdpa"] = _patched_sdpa_mask


# ═══════════════════════════════════════════════════════════════════════════
# 4. torch.diff ONNX-safe replacement
# ═══════════════════════════════════════════════════════════════════════════
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
    return (
        torch.narrow(combined, dim, 1, combined.size(dim) - 1)
        - torch.narrow(combined, dim, 0, combined.size(dim) - 1)
    )


torch.diff = _onnx_safe_diff


# ═══════════════════════════════════════════════════════════════════════════
# 5. bool cumsum ONNX fix
# ═══════════════════════════════════════════════════════════════════════════
_orig_cumsum = torch.Tensor.cumsum


def _onnx_safe_cumsum(self, dim, dtype=None):
    if self.dtype == torch.bool:
        return _orig_cumsum(self.to(torch.int64), dim, dtype=dtype)
    return _orig_cumsum(self, dim, dtype=dtype)


torch.Tensor.cumsum = _onnx_safe_cumsum


# ═══════════════════════════════════════════════════════════════════════════
# 6. Disable GQA in SDPA
# ═══════════════════════════════════════════════════════════════════════════
import transformers.integrations.sdpa_attention as _sdpa_mod

_sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False


# ═══════════════════════════════════════════════════════════════════════════
# 7. vmap-free masking registration
# ═══════════════════════════════════════════════════════════════════════════
VMAP_WORKAROUND = None

try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.executorch import sdpa_mask_without_vmap

    ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
    ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"])
    VMAP_WORKAROUND = "sdpa_without_vmap"
except ImportError:
    pass  # transformers 5.5+ — vmap-free is the default


# ═══════════════════════════════════════════════════════════════════════════
# 8. torch.cdist ONNX-safe replacement
# ═══════════════════════════════════════════════════════════════════════════
_orig_cdist = torch.cdist


def _onnx_safe_cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """ONNX-safe replacement for torch.cdist.

    The standard torch.cdist fails during ONNX trace export because
    the symbolic function can't determine row_size_x1 statically.
    This implementation uses basic tensor ops that ONNX can trace.
    """
    # x1: (..., P, M), x2: (..., R, M) → output: (..., P, R)
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 2.0:
        return (diff * diff).sum(-1).sqrt()
    elif p == 1.0:
        return diff.abs().sum(-1)
    elif p == float('inf'):
        return diff.abs().amax(-1)
    else:
        return diff.abs().pow(p).sum(-1).pow(1.0 / p)


torch.cdist = _onnx_safe_cdist


# ═══════════════════════════════════════════════════════════════════════════
# Helper: patch model attention for vmap-free ONNX export
# ═══════════════════════════════════════════════════════════════════════════
def patch_attention_for_export(model_or_layers, config=None):
    """Patch transformer layers to use vmap-free attention during ONNX export.

    Args:
        model_or_layers: Either a model with .layers attribute, or a list of layers.
        config: Optional config object to patch _attn_implementation on.

    Returns:
        The attention implementation name used ('sdpa_without_vmap', 'eager', etc.)
    """
    if VMAP_WORKAROUND is None:
        return "eager"  # transformers 5.5+ or no workaround available

    impl = VMAP_WORKAROUND

    if config is not None:
        config._attn_implementation = impl

    layers = getattr(model_or_layers, "layers", model_or_layers)
    if hasattr(layers, "__iter__"):
        for layer in layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.config._attn_implementation = impl

    return impl
