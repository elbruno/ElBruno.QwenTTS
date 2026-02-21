"""
Export Talker LM (prefill + decode) and Code Predictor to ONNX.

Creates three ONNX models:
  1. talker_prefill.onnx  — Talker LM prefill (full sequence, produces KV cache)
  2. talker_decode.onnx   — Talker LM single-step decode (with KV cache I/O)
  3. code_predictor.onnx  — Code Predictor (31-step codebook generation)

Usage:
  python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForTextToWaveform
from transformers.cache_utils import DynamicCache

# ---------------------------------------------------------------------------
# Model constants (0.6B CustomVoice defaults from config)
# ---------------------------------------------------------------------------
TALKER_NUM_LAYERS = 20
TALKER_NUM_KV_HEADS = 2
TALKER_HEAD_DIM = 128
TALKER_HIDDEN = 1024
TALKER_VOCAB = 3072

CP_NUM_LAYERS = 5
CP_NUM_KV_HEADS = 8
CP_HEAD_DIM = 128
CP_HIDDEN = 1024
CP_VOCAB = 2048
CP_NUM_GROUPS = 31  # codebook groups 1-31

OPSET_VERSION = 17


# ═══════════════════════════════════════════════════════════════════════════
# Talker LM Prefill Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TalkerPrefillWrapper(nn.Module):
    """
    Wraps the Talker LM for prefill export.

    Inputs:
      - inputs_embeds:  (B, T, 1024) float32
      - attention_mask:  (B, T) int64
      - position_ids:    (3, B, T) int64   — M-RoPE temporal/height/width

    Outputs:
      - logits:          (B, 1, 3072) float32  — last-token logits
      - hidden_states:   (B, T, 1024) float32  — full hidden for code predictor
      - present_key_values: 20 layers × [key (B,2,T,128), value (B,2,T,128)]
        flattened as: present_key_0, present_value_0, ..., present_key_19, present_value_19
    """

    def __init__(self, talker):
        super().__init__()
        self.model = talker.model  # Qwen3TTSTalkerModel
        self.codec_head = talker.codec_head

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
        for i in range(TALKER_NUM_LAYERS):
            flat_kv.append(cache.key_cache[i])
            flat_kv.append(cache.value_cache[i])

        return (logits, hidden_states, *flat_kv)


# ═══════════════════════════════════════════════════════════════════════════
# Talker LM Decode Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TalkerDecodeWrapper(nn.Module):
    """
    Wraps the Talker LM for single-token decode export.

    Inputs:
      - inputs_embeds:   (B, 1, 1024) float32
      - attention_mask:  (B, T+1) int64
      - position_ids:    (3, B, 1) int64
      - past_key_values: 20 layers × [key (B,2,T,128), value (B,2,T,128)]
        as flat args: past_key_0, past_value_0, ..., past_key_19, past_value_19

    Outputs:
      - logits:          (B, 1, 3072) float32
      - hidden_states:   (B, 1, 1024) float32
      - present_key_values: 20 layers × [key (B,2,T+1,128), value (B,2,T+1,128)]
    """

    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_kv_flat):
        # Reconstruct DynamicCache from flat KV tensors
        cache = DynamicCache()
        for i in range(TALKER_NUM_LAYERS):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

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
        flat_kv = []
        for i in range(TALKER_NUM_LAYERS):
            flat_kv.append(new_cache.key_cache[i])
            flat_kv.append(new_cache.value_cache[i])

        return (logits, hidden_states, *flat_kv)


# ═══════════════════════════════════════════════════════════════════════════
# Code Predictor Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CodePredictorWrapper(nn.Module):
    """
    Wraps the Code Predictor for ONNX export.

    The Code Predictor has 31 separate lm_head layers (one per codebook group).
    We stack all weights into a single tensor and use index_select for dynamic
    routing based on generation_steps.

    Inputs:
      - inputs_embeds:    (B, S, 1024) float32
            S=2 for prefill [talker_hidden, group_0_embed], S=1 for decode
      - generation_steps: (1,) int64  — which lm_head to use (0..30 → groups 1..31)
      - past_key_values:  5 layers × [key (B,8,S_past,128), value (B,8,S_past,128)]
            Pass zero-length (B,8,0,128) tensors for the first call.

    Outputs:
      - logits:           (B, S_out, 2048) float32  — S_out is last token's logits
      - present_key_values: 5 layers × [key, value] with updated sequence dim
    """

    def __init__(self, code_predictor):
        super().__init__()
        self.model = code_predictor.model  # Qwen3TTSTalkerCodePredictorModel
        self.projection = code_predictor.small_to_mtp_projection

        # Stack all 31 lm_head weight matrices for indexed access
        # Each lm_head[i] is nn.Linear(1024, 2048, bias=False)
        all_weights = torch.stack(
            [head.weight for head in code_predictor.lm_head]
        )  # (31, 2048, 1024)
        self.register_buffer("lm_head_weights", all_weights)

    def forward(self, inputs_embeds, generation_steps, *past_kv_flat):
        inputs_embeds = self.projection(inputs_embeds)

        # Reconstruct DynamicCache from flat KV tensors
        cache = DynamicCache()
        for i in range(CP_NUM_LAYERS):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state

        # Select the correct lm_head by generation_steps index
        weight = torch.index_select(
            self.lm_head_weights, 0, generation_steps.view(-1)
        )  # (1, 2048, 1024)
        weight = weight.squeeze(0)  # (2048, 1024)
        logits = torch.matmul(hidden_states, weight.t())  # (B, S, 2048)

        new_cache = outputs.past_key_values
        flat_kv = []
        for i in range(CP_NUM_LAYERS):
            flat_kv.append(new_cache.key_cache[i])
            flat_kv.append(new_cache.value_cache[i])

        return (logits, *flat_kv)


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


def export_talker_prefill(talker, output_dir, device):
    """Export Talker LM prefill model to ONNX."""
    print("Exporting talker_prefill.onnx ...")
    wrapper = TalkerPrefillWrapper(talker).eval().to(device)

    B, T = 1, 8  # dummy dimensions
    dummy_inputs = (
        torch.randn(B, T, TALKER_HIDDEN, device=device),
        torch.ones(B, T, dtype=torch.int64, device=device),
        torch.zeros(3, B, T, dtype=torch.int64, device=device),
    )

    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    output_names = ["logits", "hidden_states"]
    kv_out_names = _kv_output_names("present", TALKER_NUM_LAYERS)
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


def export_talker_decode(talker, output_dir, device):
    """Export Talker LM decode-step model to ONNX."""
    print("Exporting talker_decode.onnx ...")
    wrapper = TalkerDecodeWrapper(talker).eval().to(device)

    B, T_past = 1, 8  # dummy past length

    dummy_embeds = torch.randn(B, 1, TALKER_HIDDEN, device=device)
    dummy_mask = torch.ones(B, T_past + 1, dtype=torch.int64, device=device)
    dummy_pos = torch.zeros(3, B, 1, dtype=torch.int64, device=device)

    dummy_past_kv = []
    for _ in range(TALKER_NUM_LAYERS):
        dummy_past_kv.append(
            torch.randn(B, TALKER_NUM_KV_HEADS, T_past, TALKER_HEAD_DIM, device=device)
        )  # key
        dummy_past_kv.append(
            torch.randn(B, TALKER_NUM_KV_HEADS, T_past, TALKER_HEAD_DIM, device=device)
        )  # value

    dummy_inputs = (dummy_embeds, dummy_mask, dummy_pos, *dummy_past_kv)

    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    past_kv_names = _kv_input_names("past", TALKER_NUM_LAYERS)
    input_names.extend(past_kv_names)

    output_names = ["logits", "hidden_states"]
    present_kv_names = _kv_output_names("present", TALKER_NUM_LAYERS)
    output_names.extend(present_kv_names)

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {1: "batch_size"},
        "logits": {0: "batch_size"},
        "hidden_states": {0: "batch_size"},
    }
    for name in past_kv_names:
        dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
    for name in present_kv_names:
        dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}

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


def export_code_predictor(talker, output_dir, device):
    """Export Code Predictor model to ONNX."""
    print("Exporting code_predictor.onnx ...")
    wrapper = CodePredictorWrapper(talker.code_predictor).eval().to(device)

    B, S = 1, 2  # prefill: [talker_hidden, group_0_embed]
    T_past = 0   # no past for first call

    dummy_embeds = torch.randn(B, S, CP_HIDDEN, device=device)
    dummy_steps = torch.tensor([0], dtype=torch.int64, device=device)

    # Past KV: zero-length sequence for initial call
    dummy_past_kv = []
    for _ in range(CP_NUM_LAYERS):
        dummy_past_kv.append(
            torch.zeros(B, CP_NUM_KV_HEADS, T_past, CP_HEAD_DIM, device=device)
        )
        dummy_past_kv.append(
            torch.zeros(B, CP_NUM_KV_HEADS, T_past, CP_HEAD_DIM, device=device)
        )

    dummy_inputs = (dummy_embeds, dummy_steps, *dummy_past_kv)

    input_names = ["inputs_embeds", "generation_steps"]
    past_kv_names = _kv_input_names("past", CP_NUM_LAYERS)
    input_names.extend(past_kv_names)

    output_names = ["logits"]
    present_kv_names = _kv_output_names("present", CP_NUM_LAYERS)
    output_names.extend(present_kv_names)

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for name in past_kv_names:
        dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
    for name in present_kv_names:
        dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        os.path.join(output_dir, "code_predictor.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print("  ✓ code_predictor.onnx")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export Talker LM and Code Predictor to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/Qwen3-TTS-0.6B-CustomVoice",
        help="Path to the Qwen3-TTS model directory",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    model = AutoModelForTextToWaveform.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    talker = model.talker
    device = torch.device(args.device)
    talker = talker.to(device)

    with torch.no_grad():
        export_talker_prefill(talker, str(output_dir), device)
        export_talker_decode(talker, str(output_dir), device)
        export_code_predictor(talker, str(output_dir), device)

    print(f"\nAll LM exports saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
