"""
Tests for export script validation logic (Issue #34).

These tests run WITHOUT requiring model files or GPU.
They verify that export scripts catch invalid inputs early
with clear error messages instead of crashing deep in PyTorch.

Run:  cd python && python -m pytest tests/ -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add parent directory to path so we can import export_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_utils import (
    ExportValidationError,
    is_hf_repo_id,
    validate_model_dir,
    validate_repo_id,
    validate_model_config_for_lm_export,
    validate_output_dir,
    SUPPORTED_CUSTOMVOICE_REPOS,
    SUPPORTED_BASE_REPOS,
    SUPPORTED_TOKENIZER_REPOS,
    ALL_SUPPORTED_REPOS,
    KNOWN_UNSUPPORTED_REPOS,
    HF_REPO_PATTERN,
)


# ═══════════════════════════════════════════════════════════════════════════
# is_hf_repo_id tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIsHfRepoId:
    """Test detection of HuggingFace repo IDs vs local paths."""

    def test_valid_repo_id(self):
        assert is_hf_repo_id("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice") is True

    def test_valid_repo_id_with_dots(self):
        assert is_hf_repo_id("elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX") is True

    def test_local_relative_path(self):
        assert is_hf_repo_id("models/Qwen3-TTS-0.6B-CustomVoice") is False

    def test_local_absolute_unix_path(self):
        assert is_hf_repo_id("/home/user/models/Qwen3-TTS") is False

    def test_local_dot_relative_path(self):
        assert is_hf_repo_id("./models/test") is False

    def test_local_tilde_path(self):
        assert is_hf_repo_id("~/models/test") is False

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_drive_path(self):
        assert is_hf_repo_id("C:\\Users\\user\\models") is False

    def test_simple_name_not_repo(self):
        """A single name without slash is not a repo ID."""
        assert is_hf_repo_id("some-model") is False

    def test_empty_string(self):
        assert is_hf_repo_id("") is False

    def test_just_slash(self):
        assert is_hf_repo_id("/") is False

    def test_repo_with_underscores(self):
        assert is_hf_repo_id("org_name/model_name") is True

    def test_unsupported_qwen_repo(self):
        """Non-12Hz repos are still valid HF repo IDs (just not supported by scripts)."""
        assert is_hf_repo_id("Qwen/Qwen3-TTS-0.6B-CustomVoice") is True

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_backslash_path(self):
        assert is_hf_repo_id("models\\Qwen3-TTS-0.6B-CustomVoice") is False


# ═══════════════════════════════════════════════════════════════════════════
# validate_model_dir tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateModelDir:
    """Test model directory validation."""

    def test_valid_dir_with_config(self, tmp_model_dir):
        result = validate_model_dir(str(tmp_model_dir))
        assert result == tmp_model_dir.resolve()

    def test_nonexistent_dir_raises(self):
        with pytest.raises(ExportValidationError, match="does not exist"):
            validate_model_dir("/nonexistent/path/that/doesnt/exist")

    def test_empty_dir_without_config_raises(self, empty_dir):
        with pytest.raises(ExportValidationError, match="No config.json"):
            validate_model_dir(str(empty_dir))

    def test_empty_dir_no_config_required(self, empty_dir):
        """When require_config=False, empty dir is OK."""
        result = validate_model_dir(str(empty_dir), require_config=False)
        assert result == empty_dir.resolve()

    def test_file_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(ExportValidationError, match="not a directory"):
            validate_model_dir(str(f))

    def test_hf_repo_id_instead_of_path_raises(self):
        """Issue #34: User passes HF repo ID instead of local path."""
        with pytest.raises(ExportValidationError, match="looks like a HuggingFace repo ID"):
            validate_model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    def test_unsupported_hf_repo_id_raises_with_suggestion(self):
        """Issue #34: User passes non-12Hz HF repo ID — gets specific fix advice."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_dir("Qwen/Qwen3-TTS-0.6B-CustomVoice")
        err = exc_info.value
        assert "HuggingFace repo ID" in str(err)
        assert err.suggestion is not None
        assert "12Hz" in err.suggestion

    def test_supported_hf_repo_id_raises_with_download_instructions(self):
        """User passes a supported 12Hz repo — still can't use directly, need download."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_dir("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        err = exc_info.value
        assert err.suggestion is not None
        assert "download" in err.suggestion.lower()

    def test_elbruno_onnx_repo_raises(self):
        """User passes elbruno ONNX repo — wrong repo type entirely."""
        with pytest.raises(ExportValidationError, match="HuggingFace repo ID"):
            validate_model_dir("elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX")


# ═══════════════════════════════════════════════════════════════════════════
# validate_repo_id tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateRepoId:
    """Test HuggingFace repo ID validation."""

    def test_valid_customvoice_06b(self):
        result = validate_repo_id("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "lm")
        assert result == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def test_valid_customvoice_17b(self):
        result = validate_repo_id("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "lm")
        assert result == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def test_valid_base_06b(self):
        result = validate_repo_id("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "lm")
        assert result == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def test_valid_tokenizer(self):
        result = validate_repo_id("Qwen/Qwen3-TTS-Tokenizer-12Hz", "vocoder")
        assert result == "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    def test_invalid_format_no_slash(self):
        with pytest.raises(ExportValidationError, match="not a valid HuggingFace repo ID"):
            validate_repo_id("just-a-name", "lm")

    def test_invalid_format_empty(self):
        with pytest.raises(ExportValidationError, match="not a valid HuggingFace repo ID"):
            validate_repo_id("", "lm")

    def test_unsupported_non_12hz_06b_customvoice(self):
        """Issue #34 core case: user provides non-12Hz repo ID."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_repo_id("Qwen/Qwen3-TTS-0.6B-CustomVoice", "lm")
        err = exc_info.value
        assert "not supported" in str(err)
        assert "vmap" in str(err).lower() or "functorch" in str(err).lower()
        assert err.suggestion is not None

    def test_unsupported_non_12hz_17b_customvoice(self):
        """Issue #34 core case: user provides non-12Hz 1.7B repo ID."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_repo_id("Qwen/Qwen3-TTS-1.7B-CustomVoice", "lm")
        assert "not supported" in str(exc_info.value)

    def test_unsupported_non_12hz_base(self):
        with pytest.raises(ExportValidationError, match="not supported"):
            validate_repo_id("Qwen/Qwen3-TTS-0.6B-Base", "lm")

    def test_unsupported_non_12hz_17b_base(self):
        with pytest.raises(ExportValidationError, match="not supported"):
            validate_repo_id("Qwen/Qwen3-TTS-1.7B-Base", "lm")

    def test_suggestion_lists_supported_repos(self):
        """Error suggestion should list the correct repos to use."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_repo_id("Qwen/Qwen3-TTS-0.6B-CustomVoice", "lm")
        suggestion = exc_info.value.suggestion
        assert "Qwen/Qwen3-TTS-12Hz" in suggestion

    def test_unknown_repo_passes(self):
        """Unknown repos pass validation (they might be valid custom repos)."""
        result = validate_repo_id("someone/some-custom-model", "lm")
        assert result == "someone/some-custom-model"


# ═══════════════════════════════════════════════════════════════════════════
# validate_model_config_for_lm_export tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateModelConfig:
    """Test model config validation without loading actual models."""

    def _make_config(self, **overrides):
        """Create a mock config object mimicking Qwen3TTSConfig."""
        cp_config = MagicMock()
        cp_config.hidden_size = 1024
        cp_config.num_hidden_layers = 4
        cp_config.num_attention_heads = 8
        cp_config.num_key_value_heads = 4
        cp_config.vocab_size = 2048
        cp_config.head_dim = 128

        tc = MagicMock()
        tc.hidden_size = 1024
        tc.num_hidden_layers = 12
        tc.num_attention_heads = 16
        tc.num_key_value_heads = 4
        tc.vocab_size = 3072
        tc.head_dim = 128
        tc.num_code_groups = 32
        tc.code_predictor_config = cp_config

        config = MagicMock()
        config.talker_config = tc

        for key, val in overrides.items():
            setattr(config, key, val)

        return config

    def test_valid_config(self):
        config = self._make_config()
        result = validate_model_config_for_lm_export(config)
        assert result["valid"] is True

    def test_missing_talker_config(self):
        config = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(ExportValidationError, match="talker_config"):
            validate_model_config_for_lm_export(config)

    def test_missing_code_predictor_config(self):
        config = self._make_config()
        # Remove code_predictor_config
        del config.talker_config.code_predictor_config
        with pytest.raises(ExportValidationError, match="code_predictor_config"):
            validate_model_config_for_lm_export(config)

    def test_missing_talker_hidden_layers(self):
        config = self._make_config()
        del config.talker_config.num_hidden_layers
        with pytest.raises(ExportValidationError, match="num_hidden_layers"):
            validate_model_config_for_lm_export(config)

    def test_missing_num_code_groups(self):
        config = self._make_config()
        del config.talker_config.num_code_groups
        with pytest.raises(ExportValidationError, match="num_code_groups"):
            validate_model_config_for_lm_export(config)

    def test_missing_cp_vocab_size(self):
        config = self._make_config()
        del config.talker_config.code_predictor_config.vocab_size
        with pytest.raises(ExportValidationError, match="vocab_size"):
            validate_model_config_for_lm_export(config)

    def test_error_message_lists_all_missing(self):
        """Multiple missing attrs should all be listed in one error."""
        config = self._make_config()
        del config.talker_config.num_hidden_layers
        del config.talker_config.hidden_size
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_config_for_lm_export(config)
        msg = str(exc_info.value)
        assert "num_hidden_layers" in msg
        assert "hidden_size" in msg


# ═══════════════════════════════════════════════════════════════════════════
# validate_output_dir tests
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateOutputDir:
    """Test output directory validation and creation."""

    def test_existing_dir(self, output_dir):
        result = validate_output_dir(str(output_dir))
        assert result == output_dir.resolve()

    def test_creates_nested_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        result = validate_output_dir(str(nested))
        assert nested.exists()
        assert result == nested.resolve()


# ═══════════════════════════════════════════════════════════════════════════
# ExportValidationError tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExportValidationError:
    """Test the custom error class."""

    def test_message_only(self):
        err = ExportValidationError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.suggestion is None

    def test_message_with_suggestion(self):
        err = ExportValidationError("Bad input", suggestion="Try this instead")
        assert "Bad input" in str(err)
        assert "Try this instead" in str(err)
        assert "Suggestion:" in str(err)
        assert err.suggestion == "Try this instead"


# ═══════════════════════════════════════════════════════════════════════════
# Repo ID constants integrity tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRepoConstants:
    """Verify the repo ID constants are consistent and correct."""

    def test_all_supported_repos_match_hf_pattern(self):
        for repo in ALL_SUPPORTED_REPOS:
            assert HF_REPO_PATTERN.match(repo), f"{repo} doesn't match HF pattern"

    def test_all_unsupported_repos_match_hf_pattern(self):
        for repo in KNOWN_UNSUPPORTED_REPOS:
            assert HF_REPO_PATTERN.match(repo), f"{repo} doesn't match HF pattern"

    def test_no_overlap_between_supported_and_unsupported(self):
        overlap = set(ALL_SUPPORTED_REPOS) & set(KNOWN_UNSUPPORTED_REPOS)
        assert len(overlap) == 0, f"Repos in both lists: {overlap}"

    def test_supported_repos_contain_12hz(self):
        """All supported repos should contain '12Hz' in their name."""
        for repo in SUPPORTED_CUSTOMVOICE_REPOS + SUPPORTED_BASE_REPOS:
            assert "12Hz" in repo, f"{repo} missing '12Hz'"

    def test_unsupported_repos_lack_12hz(self):
        """Known unsupported repos should NOT contain '12Hz'."""
        for repo in KNOWN_UNSUPPORTED_REPOS:
            assert "12Hz" not in repo, f"{repo} has '12Hz' but is in unsupported list"

    def test_customvoice_repos_have_both_sizes(self):
        assert len(SUPPORTED_CUSTOMVOICE_REPOS) >= 2
        names = " ".join(SUPPORTED_CUSTOMVOICE_REPOS)
        assert "0.6B" in names
        assert "1.7B" in names

    def test_base_repos_have_both_sizes(self):
        assert len(SUPPORTED_BASE_REPOS) >= 2
        names = " ".join(SUPPORTED_BASE_REPOS)
        assert "0.6B" in names
        assert "1.7B" in names


# ═══════════════════════════════════════════════════════════════════════════
# Issue #34 regression tests — the exact scenarios from the bug report
# ═══════════════════════════════════════════════════════════════════════════


class TestIssue34Regression:
    """
    Regression tests for GitHub Issue #34: "export lm failed"

    The user modified export_lm.py to use official HuggingFace repo IDs
    instead of the elbruno/ repos and got:
      RuntimeError: invalid unordered_map<key>

    These tests verify that our validation catches this BEFORE PyTorch runs.
    """

    def test_official_17b_customvoice_repo_as_model_dir(self):
        """Exact scenario from issue: user sets model-dir to an HF repo ID."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_dir("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        assert "HuggingFace repo ID" in str(exc_info.value)
        assert "local directory" in str(exc_info.value).lower()

    def test_official_06b_customvoice_repo_as_model_dir(self):
        """User tries 0.6B variant."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        assert "HuggingFace repo ID" in str(exc_info.value)

    def test_non_12hz_repo_id_caught_by_validate_repo(self):
        """Non-12Hz repo IDs are caught with explanation about vmap errors."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_repo_id("Qwen/Qwen3-TTS-0.6B-CustomVoice", "lm")
        msg = str(exc_info.value)
        assert "not supported" in msg
        assert "vmap" in msg.lower() or "functorch" in msg.lower()

    def test_error_message_is_actionable(self):
        """Error messages should tell users exactly what to do instead."""
        with pytest.raises(ExportValidationError) as exc_info:
            validate_model_dir("Qwen/Qwen3-TTS-0.6B-CustomVoice")
        err = exc_info.value
        # Should have a suggestion
        assert err.suggestion is not None
        # Suggestion should mention the 12Hz variant
        assert "12Hz" in err.suggestion
        # Suggestion should include download instructions
        assert "download" in err.suggestion.lower()

    def test_all_four_non_12hz_variants_caught(self):
        """All four non-12Hz official repos should be caught."""
        non_12hz = [
            "Qwen/Qwen3-TTS-0.6B-CustomVoice",
            "Qwen/Qwen3-TTS-1.7B-CustomVoice",
            "Qwen/Qwen3-TTS-0.6B-Base",
            "Qwen/Qwen3-TTS-1.7B-Base",
        ]
        for repo in non_12hz:
            with pytest.raises(ExportValidationError, match="not supported"):
                validate_repo_id(repo, "lm")


# ═══════════════════════════════════════════════════════════════════════════
# Export script argument parsing simulation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExportScriptPatterns:
    """
    Test patterns that mimic how export scripts will call validation.
    These simulate the flow: parse args → validate → load model.
    """

    def test_export_lm_flow_with_local_dir(self, tmp_model_dir):
        """Happy path: local dir with config.json passes validation."""
        model_dir = validate_model_dir(str(tmp_model_dir))
        assert model_dir.is_dir()
        assert (model_dir / "config.json").exists()

    def test_export_lm_flow_with_hf_repo_id(self):
        """Sad path: HF repo ID instead of local path."""
        with pytest.raises(ExportValidationError):
            validate_model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    def test_export_embeddings_flow_with_local_dir(self, tmp_model_dir):
        """export_embeddings.py happy path."""
        model_dir = validate_model_dir(str(tmp_model_dir))
        assert (model_dir / "config.json").exists()

    def test_export_vocoder_flow_with_repo_id(self):
        """export_vocoder.py uses --model-dir which can be a local path."""
        # When using None (default), the script downloads from HF
        # When using a local path, it should work
        # When using a repo ID, it should fail
        with pytest.raises(ExportValidationError):
            validate_model_dir("Qwen/Qwen3-TTS-Tokenizer-12Hz")

    def test_output_dir_creation(self, tmp_path):
        """Output dir should be created automatically."""
        new_dir = tmp_path / "onnx_output" / "lm"
        result = validate_output_dir(str(new_dir))
        assert new_dir.exists()
