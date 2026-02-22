"""
Export ECAPA-TDNN Speaker Encoder to ONNX.

Creates:
  speaker_encoder.onnx — Speaker encoder for voice cloning (Base model only)

Input:  mel_spectrogram (1, T_mel, 128) float32  — 128-dim mel-spectrogram (time-first)
Output: speaker_embedding (1, 1024) float32       — x-vector speaker embedding

Usage:
  python export_speaker_encoder.py --model-dir models/Qwen3-TTS-0.6B-Base --output-dir onnx_base/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig


OPSET_VERSION = 17


class SpeakerEncoderWrapper(nn.Module):
    """Wraps the ECAPA-TDNN speaker encoder for clean ONNX export."""

    def __init__(self, speaker_encoder):
        super().__init__()
        self.encoder = speaker_encoder

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: (B, T_mel, n_mels) float32 — time-first layout
        Returns:
            speaker_embedding: (B, enc_dim) float32
        """
        return self.encoder(mel_spectrogram)


def export_speaker_encoder(model_dir: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...")
    config = Qwen3TTSConfig.from_pretrained(model_dir)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, config=config, dtype=torch.float32
    )
    model.eval()

    if not hasattr(model, "speaker_encoder") or model.speaker_encoder is None:
        raise RuntimeError(
            "This model does not have a speaker encoder. "
            "Use the Base model (not CustomVoice) for voice cloning."
        )

    se = model.speaker_encoder
    wrapper = SpeakerEncoderWrapper(se)
    wrapper.eval()

    # Dummy input: (batch=1, T_mel=300, n_mels=128) — ~2.5s of audio at 24kHz/256 hop
    # The encoder internally transposes to (B, n_mels, T_mel) for Conv1d
    n_mels = config.speaker_encoder_config.mel_dim
    dummy_mel = torch.randn(1, 300, n_mels)

    # Run PyTorch reference
    with torch.no_grad():
        ref_output = wrapper(dummy_mel)
    print(f"PyTorch output shape: {ref_output.shape}")

    # Export to ONNX
    onnx_path = output_path / "speaker_encoder.onnx"
    print(f"\nExporting to {onnx_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(onnx_path),
        opset_version=OPSET_VERSION,
        input_names=["mel_spectrogram"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 1: "time"},
            "speaker_embedding": {0: "batch"},
        },
    )
    print(f"  ✓ Exported {onnx_path.name}")

    # Validate ONNX model
    print("\nValidating ONNX model ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid")

    # Compare outputs
    sess = ort.InferenceSession(str(onnx_path))
    ort_output = sess.run(None, {"mel_spectrogram": dummy_mel.numpy()})[0]
    ort_tensor = torch.from_numpy(ort_output)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), ort_tensor.flatten(), dim=0
    ).item()

    # Max absolute error
    max_err = (ref_output - ort_tensor).abs().max().item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute error: {max_err:.6e}")

    if cos_sim > 0.999:
        print("  ✓ PASS: Cosine similarity > 0.999")
    else:
        print("  ✗ FAIL: Cosine similarity too low!")

    # Test with different sequence lengths
    print("\nTesting dynamic time axis ...")
    for t in [100, 500, 1000]:
        test_mel = torch.randn(1, t, n_mels)
        with torch.no_grad():
            pt_out = wrapper(test_mel)
        ort_out = sess.run(None, {"mel_spectrogram": test_mel.numpy()})[0]
        err = np.abs(pt_out.numpy() - ort_out).max()
        print(f"  T={t:4d}: shape={ort_out.shape}, max_err={err:.6e}")

    print(f"\nSpeaker encoder exported to {onnx_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Export ECAPA-TDNN speaker encoder to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/Qwen3-TTS-0.6B-Base",
        help="Path to the Qwen3-TTS Base model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_base",
        help="Directory to save speaker_encoder.onnx",
    )
    args = parser.parse_args()
    export_speaker_encoder(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
