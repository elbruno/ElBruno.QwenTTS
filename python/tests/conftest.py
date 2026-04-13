"""Pytest configuration and shared fixtures for export script tests."""

import json
import os
import shutil
import tempfile

import pytest


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory simulating a valid model directory."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Minimal config.json that resembles a Qwen3-TTS model
    config = {
        "model_type": "qwen3_tts",
        "talker_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 3072,
            "head_dim": 128,
            "num_code_groups": 32,
            "text_hidden_size": 2048,
            "codec_eos_token_id": 2048,
            "codec_think_id": 2049,
            "codec_nothink_id": 2050,
            "codec_think_bos_id": 2051,
            "codec_think_eos_id": 2052,
            "codec_pad_id": 2053,
            "codec_bos_id": 2054,
            "rope_theta": 1000000.0,
            "code_predictor_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "vocab_size": 2048,
                "head_dim": 128,
                "rope_theta": 1000000.0,
            },
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    return model_dir


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty temporary directory."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory for ONNX files."""
    d = tmp_path / "onnx_output"
    d.mkdir()
    return d
