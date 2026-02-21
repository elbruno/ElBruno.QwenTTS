"""
Extract embedding weights and projection layers as NumPy arrays.

Saves raw tensors that the C# runtime loads directly — no ONNX needed
for these since they are pure lookup tables and small linear layers.

Outputs (all .npy):
  embeddings/
    text_embedding.npy              — (text_vocab, 2048) float32
    text_projection_fc1_weight.npy  — (2048, 2048) float32
    text_projection_fc1_bias.npy    — (2048,) float32
    text_projection_fc2_weight.npy  — (1024, 2048) float32
    text_projection_fc2_bias.npy    — (1024,) float32
    talker_codec_embedding.npy      — (3072, 1024) float32  (group 0 + control tokens)
    cp_codec_embedding_{0..30}.npy  — (2048, 1024) float32  (groups 1-31)
    codec_head_weight.npy           — (3072, 1024) float32
    speaker_ids.json                — {"speaker_name": [id0, id1, ...], ...}
    config.json                     — special token IDs and model dimensions

Usage:
  python export_embeddings.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice --output-dir onnx/embeddings
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig


def save_tensor(tensor, path):
    """Save a PyTorch tensor as a NumPy .npy file."""
    np.save(str(path), tensor.detach().cpu().float().numpy())
    print(f"  ✓ {path.name}  {tuple(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embedding weights as NumPy arrays for C# inference"
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
        default="onnx/embeddings",
        help="Directory to save .npy embedding files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    config = Qwen3TTSConfig.from_pretrained(args.model_dir)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.float32,
    )
    model.eval()

    talker = model.talker
    config = model.config
    talker_config = config.talker_config

    # ── Text embedding (BPE vocab → 2048-dim) ────────────────────────────
    print("\nText embedding:")
    text_emb = talker.model.text_embedding.weight  # (text_vocab, text_hidden_size)
    save_tensor(text_emb, output_dir / "text_embedding.npy")

    # ── Text projection (2048 → 1024 via SiLU MLP) ───────────────────────
    print("\nText projection (ResizeMLP):")
    proj = talker.text_projection
    save_tensor(proj.linear_fc1.weight, output_dir / "text_projection_fc1_weight.npy")
    save_tensor(proj.linear_fc1.bias, output_dir / "text_projection_fc1_bias.npy")
    save_tensor(proj.linear_fc2.weight, output_dir / "text_projection_fc2_weight.npy")
    save_tensor(proj.linear_fc2.bias, output_dir / "text_projection_fc2_bias.npy")

    # ── Talker codec embedding (group 0 + control tokens) ────────────────
    # This embedding table covers codec tokens (0..2047) AND special tokens
    # (EOS, think, pad, bos, speaker IDs, language IDs, etc. up to 3071).
    print("\nTalker codec embedding (group 0):")
    codec_emb = talker.model.codec_embedding.weight  # (3072, 1024)
    save_tensor(codec_emb, output_dir / "talker_codec_embedding.npy")

    # ── Code Predictor codec embeddings (groups 1-31) ─────────────────────
    # 31 separate embedding tables, each (2048, 1024).
    # cp_codec_embedding_0 → codebook group 1
    # cp_codec_embedding_30 → codebook group 31
    print("\nCode Predictor codec embeddings (31 tables):")
    cp_embeddings = talker.code_predictor.model.codec_embedding
    for i, emb in enumerate(cp_embeddings):
        save_tensor(emb.weight, output_dir / f"cp_codec_embedding_{i}.npy")

    # ── Codec head (Talker output projection) ─────────────────────────────
    print("\nCodec head:")
    codec_head_w = talker.codec_head.weight  # (3072, 1024)
    save_tensor(codec_head_w, output_dir / "codec_head_weight.npy")

    # ── Speaker ID mapping ────────────────────────────────────────────────
    # For CustomVoice, speakers map to token IDs in the Talker codec embedding.
    # The embedding for speaker X is: talker_codec_embedding[spk_id[X]]
    print("\nSpeaker IDs:")
    spk_ids = talker_config.spk_id or {}
    with open(output_dir / "speaker_ids.json", "w") as f:
        json.dump(spk_ids, f, indent=2)
    print(f"  ✓ speaker_ids.json  ({len(spk_ids)} speakers)")

    # ── Config metadata for C# runtime ────────────────────────────────────
    print("\nConfig:")
    meta = {
        "talker": {
            "hidden_size": talker_config.hidden_size,
            "text_hidden_size": talker_config.text_hidden_size,
            "vocab_size": talker_config.vocab_size,
            "num_hidden_layers": talker_config.num_hidden_layers,
            "num_attention_heads": talker_config.num_attention_heads,
            "num_key_value_heads": talker_config.num_key_value_heads,
            "head_dim": getattr(talker_config, "head_dim", 128),
            "num_code_groups": talker_config.num_code_groups,
            "codec_eos_token_id": talker_config.codec_eos_token_id,
            "codec_think_id": talker_config.codec_think_id,
            "codec_nothink_id": talker_config.codec_nothink_id,
            "codec_think_bos_id": talker_config.codec_think_bos_id,
            "codec_think_eos_id": talker_config.codec_think_eos_id,
            "codec_pad_id": talker_config.codec_pad_id,
            "codec_bos_id": talker_config.codec_bos_id,
            "rope_theta": talker_config.rope_theta,
        },
        "code_predictor": {
            "hidden_size": talker_config.code_predictor_config.hidden_size,
            "vocab_size": talker_config.code_predictor_config.vocab_size,
            "num_hidden_layers": talker_config.code_predictor_config.num_hidden_layers,
            "num_attention_heads": talker_config.code_predictor_config.num_attention_heads,
            "num_key_value_heads": talker_config.code_predictor_config.num_key_value_heads,
            "head_dim": getattr(talker_config.code_predictor_config, "head_dim", 128),
            "rope_theta": talker_config.code_predictor_config.rope_theta,
        },
        "tts": {
            "tts_bos_token_id": config.tts_bos_token_id,
            "tts_eos_token_id": config.tts_eos_token_id,
            "tts_pad_token_id": config.tts_pad_token_id,
            "im_start_token_id": config.im_start_token_id,
            "im_end_token_id": config.im_end_token_id,
        },
        "language_ids": talker_config.codec_language_id or {},
        "speaker_dialect": talker_config.spk_is_dialect or {},
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ config.json")

    print(f"\nAll embeddings saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
