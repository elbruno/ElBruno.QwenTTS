"""
Export Talker LM (prefill + decode) and Code Predictor to ONNX.

Creates three ONNX models:
  1. talker_prefill.onnx  — Talker LM prefill (full sequence, produces KV cache)
  2. talker_decode.onnx   — Talker LM single-step decode (with KV cache I/O)
  3. code_predictor.onnx  — Code Predictor (codebook generation)

All dimensions are read from the model config, so both 0.6B and 1.7B variants
are supported without code changes.

Usage:
  python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/
  python export_lm.py --model-dir models/Qwen3-TTS-1.7B-CustomVoice --output-dir onnx_1.7b/

Requirements:
  pip install -r requirements.txt
  Requires: qwen-tts, torch, transformers>=4.57.3, onnx, optimum[onnx]

Note:
  This script includes compatibility patches for transformers 4.57+/5.5+ that
  fix vmap-based masking crashes during ONNX tracing. If you see errors like
  "RuntimeError: invalid unordered_map<K, T> key", ensure compat_patches.py
  is present in the same directory.
"""

import argparse
import os
import sys
from pathlib import Path

from export_utils import configure_output_encoding

# Apply compatibility patches BEFORE importing qwen_tts or model code.
# These fix vmap masking, RoPE init, sdpa_mask, torch.diff, and other
# incompatibilities between qwen_tts and newer transformers versions.
try:
    import compat_patches  # noqa: F401
except ImportError:
    print("WARNING: compat_patches.py not found. Export may fail with newer transformers.")
    print("         Ensure compat_patches.py is in the same directory as this script.")

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
except ImportError:
    print("ERROR: qwen-tts package not found.")
    print("       Install with: pip install qwen-tts")
    print("       Or:           pip install -r requirements.txt")
    sys.exit(1)

OPSET_VERSION = 17


def read_model_dims(config):
    """Extract export-relevant dimensions from model config.

    Works for any Qwen3-TTS model variant (0.6B, 1.7B, etc.)."""
    tc = config.talker_config
    cp = tc.code_predictor_config
    dims = {
        "talker_num_layers": tc.num_hidden_layers,
        "talker_num_kv_heads": tc.num_key_value_heads,
        "talker_head_dim": getattr(tc, "head_dim", 128),
        "talker_hidden": tc.hidden_size,
        "talker_vocab": tc.vocab_size,
        "cp_num_layers": cp.num_hidden_layers,
        "cp_num_kv_heads": cp.num_key_value_heads,
        "cp_head_dim": getattr(cp, "head_dim", 128),
        "cp_hidden": cp.hidden_size,
        "cp_vocab": cp.vocab_size,
        "cp_num_groups": tc.num_code_groups - 1,
    }
    print(f"  Talker:  hidden={dims['talker_hidden']}, layers={dims['talker_num_layers']}, "
          f"kv_heads={dims['talker_num_kv_heads']}, head_dim={dims['talker_head_dim']}")
    print(f"  CodePred: hidden={dims['cp_hidden']}, layers={dims['cp_num_layers']}, "
          f"kv_heads={dims['cp_num_kv_heads']}, groups={dims['cp_num_groups']}")
    return dims


# ═══════════════════════════════════════════════════════════════════════════
# Talker LM Prefill Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TalkerPrefillWrapper(nn.Module):
    """Wraps the Talker LM for prefill export.

    Dimensions vary by model variant (0.6B: hidden=1024, 1.7B: hidden=2048).
    """

    def __init__(self, talker, num_layers):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states[:, -1:, :])

        cache = outputs.past_key_values
        flat_kv = []
        for i in range(self.num_layers):
            flat_kv.append(cache.layers[i].keys)
            flat_kv.append(cache.layers[i].values)

        return (logits, hidden_states, *flat_kv)


# ═══════════════════════════════════════════════════════════════════════════
# Talker LM Decode Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TalkerDecodeWrapper(nn.Module):
    """Wraps the Talker LM for single-token decode export."""

    def __init__(self, talker, num_layers):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, past_keys, past_values):
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(self.num_layers)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(self.num_layers)])

        return (logits, hidden_states, present_keys, present_values)


# ═══════════════════════════════════════════════════════════════════════════
# Code Predictor Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CodePredictorWrapper(nn.Module):
    """Wraps the Code Predictor for ONNX export.

    Stacks all lm_head weights into a single tensor for indexed access.
    """

    def __init__(self, code_predictor, num_layers):
        super().__init__()
        self.model = code_predictor.model
        # NOTE: small_to_mtp_projection is NOT stored here — it's exported
        # separately as .npy files via export_embeddings.py. Storing it as
        # a submodule would leak its weights into the ONNX graph.
        self.num_layers = num_layers

        all_weights = torch.stack(
            [head.weight for head in code_predictor.lm_head]
        )
        self.register_buffer("lm_head_weights", all_weights)

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        # Projection is NOT applied here — C# applies it externally.
        # For 1.7B (where cp_codec_embedding dim > cp_hidden), C# must project
        # ALL CP inputs (prefill AND decode steps) from talker space to CP space.

        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state

        weight = torch.index_select(
            self.lm_head_weights, 0, generation_steps.view(-1)
        )
        weight = weight.squeeze(0)
        logits = torch.matmul(hidden_states, weight.t())

        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(self.num_layers)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(self.num_layers)])

        return (logits, present_keys, present_values)


# ═══════════════════════════════════════════════════════════════════════════
# ONNX export helpers
# ═══════════════════════════════════════════════════════════════════════════

def _kv_input_names(prefix, num_layers):
    """Generate input names for past KV cache tensors."""
    names = []
    for i in range(num_layers):
        names.append(f"{prefix}_key_{i}")
        names.append(f"{prefix}_value_{i}")
    return names


def _kv_output_names(prefix, num_layers):
    """Generate output names for present KV cache tensors."""
    names = []
    for i in range(num_layers):
        names.append(f"{prefix}_key_{i}")
        names.append(f"{prefix}_value_{i}")
    return names


def _kv_dynamic_axes(names, batch_dim=0, seq_dim=2):
    """Generate dynamic axes for KV cache tensors (batch + sequence dims)."""
    axes = {}
    for name in names:
        axes[name] = {batch_dim: "batch_size", seq_dim: "past_sequence_length"}
    return axes


def export_talker_prefill(talker, output_dir, device, dims):
    """Export Talker LM prefill model to ONNX."""
    print("Exporting talker_prefill.onnx ...")
    wrapper = TalkerPrefillWrapper(talker, dims["talker_num_layers"]).eval().to(device)

    B, T = 1, 8  # dummy dimensions
    dummy_inputs = (
        torch.randn(B, T, dims["talker_hidden"], device=device),
        torch.ones(B, T, dtype=torch.int64, device=device),
        torch.zeros(3, B, T, dtype=torch.int64, device=device),
    )

    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    output_names = ["logits", "hidden_states"]
    kv_out_names = _kv_output_names("present", dims["talker_num_layers"])
    output_names.extend(kv_out_names)

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "position_ids": {1: "batch_size", 2: "sequence_length"},
        "logits": {0: "batch_size"},
        "hidden_states": {0: "batch_size", 1: "sequence_length"},
    }
    # KV outputs have dynamic batch and sequence dims
    for name in kv_out_names:
        dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        os.path.join(output_dir, "talker_prefill.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print("  ✓ talker_prefill.onnx")


def export_talker_decode(talker, output_dir, device, dims):
    """Export Talker LM decode-step model to ONNX."""
    print("Exporting talker_decode.onnx ...")
    wrapper = TalkerDecodeWrapper(talker, dims["talker_num_layers"]).eval().to(device)

    B, T_past = 1, 8  # dummy past length

    dummy_embeds = torch.randn(B, 1, dims["talker_hidden"], device=device)
    dummy_mask = torch.ones(B, T_past + 1, dtype=torch.int64, device=device)
    dummy_pos = torch.zeros(3, B, 1, dtype=torch.int64, device=device)

    # Stacked KV cache: (num_layers, B, num_kv_heads, T, head_dim)
    dummy_past_keys = torch.randn(
        dims["talker_num_layers"], B, dims["talker_num_kv_heads"],
        T_past, dims["talker_head_dim"], device=device
    )
    dummy_past_values = torch.randn(
        dims["talker_num_layers"], B, dims["talker_num_kv_heads"],
        T_past, dims["talker_head_dim"], device=device
    )

    dummy_inputs = (dummy_embeds, dummy_mask, dummy_pos, dummy_past_keys, dummy_past_values)

    input_names = ["inputs_embeds", "attention_mask", "position_ids", "past_keys", "past_values"]
    output_names = ["logits", "hidden_states", "present_keys", "present_values"]

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {1: "batch_size"},
        "logits": {0: "batch_size"},
        "hidden_states": {0: "batch_size"},
        "past_keys": {1: "batch_size", 3: "past_sequence_length"},
        "past_values": {1: "batch_size", 3: "past_sequence_length"},
        "present_keys": {1: "batch_size", 3: "total_sequence_length"},
        "present_values": {1: "batch_size", 3: "total_sequence_length"},
    }

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        os.path.join(output_dir, "talker_decode.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print("  ✓ talker_decode.onnx")


def export_code_predictor(talker, output_dir, device, dims):
    """Export Code Predictor model to ONNX."""
    print("Exporting code_predictor.onnx ...")
    wrapper = CodePredictorWrapper(talker.code_predictor, dims["cp_num_layers"]).eval().to(device)

    B, S = 1, 2  # prefill: [talker_hidden, group_0_embed]
    T_past = 2   # small past length for export tracing

    dummy_embeds = torch.randn(B, S, dims["cp_hidden"], device=device)
    dummy_steps = torch.tensor([0], dtype=torch.int64, device=device)

    # Stacked KV cache: (num_layers, B, num_kv_heads, T_past, head_dim)
    dummy_past_keys = torch.randn(
        dims["cp_num_layers"], B, dims["cp_num_kv_heads"],
        T_past, dims["cp_head_dim"], device=device
    )
    dummy_past_values = torch.randn(
        dims["cp_num_layers"], B, dims["cp_num_kv_heads"],
        T_past, dims["cp_head_dim"], device=device
    )

    dummy_inputs = (dummy_embeds, dummy_steps, dummy_past_keys, dummy_past_values)

    input_names = ["inputs_embeds", "generation_steps", "past_keys", "past_values"]
    output_names = ["logits", "present_keys", "present_values"]

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
        "past_keys": {1: "batch_size", 3: "past_sequence_length"},
        "past_values": {1: "batch_size", 3: "past_sequence_length"},
        "present_keys": {1: "batch_size", 3: "total_sequence_length"},
        "present_values": {1: "batch_size", 3: "total_sequence_length"},
    }

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        os.path.join(output_dir, "code_predictor.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )
    print("  ✓ code_predictor.onnx")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    configure_output_encoding()
    parser = argparse.ArgumentParser(
        description="Export Talker LM and Code Predictor to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/Qwen3-TTS-0.6B-CustomVoice",
        help="Path to the local Qwen3-TTS model directory (download first with download_models.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx",
        help="Directory to save ONNX files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export",
    )
    args = parser.parse_args()

    # Validate model directory — catch common mistake of passing HuggingFace
    # repo IDs instead of local paths
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        if "/" in model_dir and not os.path.exists(model_dir):
            print(f"\n  It looks like '{model_dir}' might be a HuggingFace repo ID.")
            print("  This script requires a LOCAL directory with downloaded model weights.")
            print("\n  To download models first, run:")
            print("    python download_models.py")
            print(f"\n  Then export with:")
            print(f"    python export_lm.py --model-dir models/<model-name> --output-dir {args.output_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...")
    config = Qwen3TTSConfig.from_pretrained(model_dir)

    print("Model dimensions (from config):")
    dims = read_model_dims(config)

    # Use vmap-free masking for ONNX trace compatibility.
    # The vmap-based masking in transformers 4.57+ crashes during
    # torch.onnx.export with: RuntimeError: invalid unordered_map<K, T> key
    attn_impl = "sdpa"
    try:
        from compat_patches import VMAP_WORKAROUND, patch_attention_for_export
        if VMAP_WORKAROUND:
            attn_impl = "sdpa"  # will be patched to vmap-free after loading
            print(f"  Using vmap-free attention: {VMAP_WORKAROUND}")
        else:
            attn_impl = "eager"
            print("  Using eager attention (transformers 5.5+ detected)")
    except ImportError:
        attn_impl = "eager"
        print("  Using eager attention (compat_patches not available)")

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.float32,
        attn_implementation=attn_impl,
    )
    model.eval()

    # Patch attention for vmap-free ONNX export
    try:
        from compat_patches import patch_attention_for_export
        # Patch Talker LM layers
        patch_attention_for_export(
            model.talker.model, config=model.talker.model.config
        )
        # Patch Code Predictor layers
        patch_attention_for_export(
            model.talker.code_predictor.model,
            config=model.talker.code_predictor.model.config,
        )
        print("  Patched attention layers for ONNX export")
    except ImportError:
        pass

    talker = model.talker
    device = torch.device(args.device)
    talker = talker.to(device)

    with torch.no_grad():
        export_talker_prefill(talker, str(output_dir), device, dims)
        export_talker_decode(talker, str(output_dir), device, dims)
        export_code_predictor(talker, str(output_dir), device, dims)

    print(f"\nAll LM exports saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
