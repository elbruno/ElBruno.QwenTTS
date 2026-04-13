"""
Integration tests for export scripts that require model files.

These tests are SKIPPED by default — they require multi-GB model downloads.
Run with: python -m pytest tests/test_export_integration.py -v --run-integration

To set up:
  python download_models.py --model customvoice
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from export_utils import validate_model_dir, validate_model_config_for_lm_export

# Skip all tests in this file unless --run-integration is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require model files. Set RUN_INTEGRATION_TESTS=1 to run.",
)

MODELS_DIR = Path(__file__).parent.parent / "models"


class TestExportLmIntegration:
    """Integration tests for export_lm.py with real model files."""

    @pytest.mark.skipif(
        not (MODELS_DIR / "Qwen3-TTS-0.6B-CustomVoice").exists(),
        reason="0.6B CustomVoice model not downloaded",
    )
    def test_validate_06b_customvoice_config(self):
        """Validate that the real 0.6B model config passes validation."""
        model_dir = str(MODELS_DIR / "Qwen3-TTS-0.6B-CustomVoice")
        path = validate_model_dir(model_dir)
        assert path.exists()

        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        config = Qwen3TTSConfig.from_pretrained(model_dir)
        result = validate_model_config_for_lm_export(config)
        assert result["valid"] is True

    @pytest.mark.skipif(
        not (MODELS_DIR / "Qwen3-TTS-1.7B-CustomVoice").exists(),
        reason="1.7B CustomVoice model not downloaded",
    )
    def test_validate_17b_customvoice_config(self):
        """Validate that the real 1.7B model config passes validation."""
        model_dir = str(MODELS_DIR / "Qwen3-TTS-1.7B-CustomVoice")
        path = validate_model_dir(model_dir)
        assert path.exists()

        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        config = Qwen3TTSConfig.from_pretrained(model_dir)
        result = validate_model_config_for_lm_export(config)
        assert result["valid"] is True

    @pytest.mark.skipif(
        not (MODELS_DIR / "Qwen3-TTS-0.6B-CustomVoice").exists(),
        reason="0.6B CustomVoice model not downloaded",
    )
    def test_06b_model_dimensions(self):
        """Verify expected dimensions for the 0.6B model."""
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        model_dir = str(MODELS_DIR / "Qwen3-TTS-0.6B-CustomVoice")
        config = Qwen3TTSConfig.from_pretrained(model_dir)

        assert config.talker_config.hidden_size == 1024
        assert config.talker_config.num_hidden_layers == 12
        assert config.talker_config.code_predictor_config.hidden_size == 1024

    @pytest.mark.skipif(
        not (MODELS_DIR / "Qwen3-TTS-1.7B-CustomVoice").exists(),
        reason="1.7B CustomVoice model not downloaded",
    )
    def test_17b_model_dimensions(self):
        """Verify expected dimensions for the 1.7B model (larger hidden, needs projection)."""
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        model_dir = str(MODELS_DIR / "Qwen3-TTS-1.7B-CustomVoice")
        config = Qwen3TTSConfig.from_pretrained(model_dir)

        assert config.talker_config.hidden_size == 2048
        assert config.talker_config.code_predictor_config.hidden_size == 1024
