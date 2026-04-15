"""
Shared validation utilities for Qwen3-TTS export scripts.

Provides input validation, repo ID checking, and model directory verification
to catch user errors early with clear messages instead of cryptic PyTorch errors.

Used by: export_lm.py, export_embeddings.py, export_vocoder.py,
         export_speech_tokenizer.py, export_speaker_encoder.py
"""

import json
import os
import re
import sys
from pathlib import Path


def configure_output_encoding():
    """Ensure stdout/stderr can handle Unicode on all platforms (e.g., GBK Windows)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# Supported model repo patterns
# ═══════════════════════════════════════════════════════════════════════════

# These are the HuggingFace repo IDs that contain the correct model architecture
# for our export scripts. The scripts rely on qwen_tts.core.models which expect
# specific config attributes (talker_config, code_predictor_config, etc.).
SUPPORTED_CUSTOMVOICE_REPOS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]

SUPPORTED_BASE_REPOS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]

SUPPORTED_TOKENIZER_REPOS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]

ALL_SUPPORTED_REPOS = (
    SUPPORTED_CUSTOMVOICE_REPOS
    + SUPPORTED_BASE_REPOS
    + SUPPORTED_TOKENIZER_REPOS
)

# Repo IDs that look like Qwen TTS but are NOT the 12Hz variants our scripts expect.
# These use different architectures and will fail with confusing errors.
KNOWN_UNSUPPORTED_REPOS = [
    "Qwen/Qwen3-TTS-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-0.6B-Base",
    "Qwen/Qwen3-TTS-1.7B-Base",
]

# Pattern for valid HuggingFace repo IDs: org/model-name
HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


# ═══════════════════════════════════════════════════════════════════════════
# Validation functions
# ═══════════════════════════════════════════════════════════════════════════

class ExportValidationError(Exception):
    """Raised when export script input validation fails."""

    def __init__(self, message: str, suggestion: str | None = None):
        self.suggestion = suggestion
        full_msg = message
        if suggestion:
            full_msg += f"\n\nSuggestion: {suggestion}"
        super().__init__(full_msg)


def is_hf_repo_id(path_or_repo: str) -> bool:
    """Check if a string looks like a HuggingFace repo ID (org/model) vs a local path."""
    if os.path.sep in path_or_repo or (os.name == "nt" and "\\" in path_or_repo):
        return False
    if path_or_repo.startswith((".", "/", "~")):
        return False
    if ":" in path_or_repo and len(path_or_repo) > 1 and path_or_repo[1] == ":":
        return False  # Windows drive letter (C:\...)
    if not HF_REPO_PATTERN.match(path_or_repo):
        return False
    # If the "org" part looks like a filesystem path component, it's probably a path.
    # Real HF orgs are like "Qwen", "elbruno", "meta-llama" — not "models", "src", etc.
    org = path_or_repo.split("/")[0]
    path_like_prefixes = {
        "models", "model", "src", "python", "data", "output", "outputs",
        "onnx", "onnx_models", "onnx_runtime", "weights", "checkpoints",
        "tmp", "temp", "build", "dist", "lib", "bin", "var", "etc",
        "home", "usr", "opt",
    }
    if org.lower() in path_like_prefixes:
        return False
    return True


def validate_model_dir(model_dir: str, require_config: bool = True) -> Path:
    """Validate that model_dir is a valid local directory with model files.

    Args:
        model_dir: Path to the model directory (local path or HF repo ID).
        require_config: If True, check for config.json in the directory.

    Returns:
        Resolved Path to the model directory.

    Raises:
        ExportValidationError: If the directory is invalid.
    """
    # Check if user accidentally passed a HuggingFace repo ID
    if is_hf_repo_id(model_dir):
        suggestion = _suggest_for_repo_id(model_dir)
        raise ExportValidationError(
            f"'{model_dir}' looks like a HuggingFace repo ID, not a local directory.\n"
            f"The export scripts require locally downloaded model files.",
            suggestion=suggestion,
        )

    path = Path(model_dir)

    if not path.exists():
        raise ExportValidationError(
            f"Model directory does not exist: {path.resolve()}\n"
            f"Download the model first with:\n"
            f"  python download_models.py --model customvoice",
        )

    if not path.is_dir():
        raise ExportValidationError(
            f"Model path is not a directory: {path.resolve()}",
        )

    if require_config:
        config_path = path / "config.json"
        if not config_path.exists():
            raise ExportValidationError(
                f"No config.json found in {path.resolve()}\n"
                f"This directory does not appear to contain a valid Qwen3-TTS model.",
                suggestion="Ensure you downloaded the full model, not just weights.",
            )

    return path.resolve()


def validate_repo_id(repo_id: str, script_type: str = "lm") -> str:
    """Validate a HuggingFace repo ID for use with export scripts.

    Args:
        repo_id: The HuggingFace repo ID to validate.
        script_type: Which export script is calling — "lm", "embeddings",
                     "vocoder", "speech_tokenizer", "speaker_encoder".

    Returns:
        The validated repo_id if supported.

    Raises:
        ExportValidationError: If the repo ID is not supported.
    """
    if not HF_REPO_PATTERN.match(repo_id):
        raise ExportValidationError(
            f"'{repo_id}' is not a valid HuggingFace repo ID.\n"
            f"Expected format: org/model-name (e.g., Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)",
        )

    # Check for known unsupported repos (the non-12Hz variants)
    if repo_id in KNOWN_UNSUPPORTED_REPOS:
        supported = _get_supported_for_type(script_type)
        raise ExportValidationError(
            f"Repo '{repo_id}' is not supported by the export scripts.\n"
            f"The non-12Hz Qwen3-TTS models use a different architecture that\n"
            f"is incompatible with ONNX export (causes vmap/functorch errors).",
            suggestion=f"Use one of these repos instead:\n  "
            + "\n  ".join(supported),
        )

    return repo_id


def validate_model_config_for_lm_export(config) -> dict:
    """Validate that a loaded model config has the attributes needed for LM export.

    Args:
        config: A Qwen3TTSConfig object loaded from from_pretrained().

    Returns:
        Dict with validated dimension info.

    Raises:
        ExportValidationError: If required config attributes are missing.
    """
    errors = []

    if not hasattr(config, "talker_config"):
        errors.append("Missing 'talker_config' — this model may not be a Qwen3-TTS model.")

    if hasattr(config, "talker_config"):
        tc = config.talker_config
        required_attrs = [
            "num_hidden_layers",
            "num_key_value_heads",
            "hidden_size",
            "vocab_size",
        ]
        for attr in required_attrs:
            if not hasattr(tc, attr):
                errors.append(f"Missing talker_config.{attr}")

        if hasattr(tc, "code_predictor_config"):
            cp = tc.code_predictor_config
            for attr in required_attrs:
                if not hasattr(cp, attr):
                    errors.append(f"Missing talker_config.code_predictor_config.{attr}")
        else:
            errors.append("Missing 'talker_config.code_predictor_config'")

        if not hasattr(tc, "num_code_groups"):
            errors.append("Missing 'talker_config.num_code_groups'")

    if errors:
        raise ExportValidationError(
            "Model config is missing required attributes for LM export:\n  "
            + "\n  ".join(errors),
            suggestion="Ensure you are using a Qwen3-TTS model (0.6B or 1.7B CustomVoice/Base).",
        )

    return {"valid": True}


def validate_output_dir(output_dir: str) -> Path:
    """Validate and create the output directory for exported models.

    Args:
        output_dir: Path for ONNX output files.

    Returns:
        Resolved Path to the output directory.
    """
    path = Path(output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ExportValidationError(
            f"Cannot create output directory '{path.resolve()}': {e}",
        )
    return path.resolve()


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _suggest_for_repo_id(repo_id: str) -> str:
    """Generate a helpful suggestion when user provides a repo ID instead of a path."""
    if repo_id in KNOWN_UNSUPPORTED_REPOS:
        # Map non-12Hz to 12Hz
        corrected = repo_id.replace("Qwen/Qwen3-TTS-", "Qwen/Qwen3-TTS-12Hz-")
        return (
            f"The repo '{repo_id}' uses a non-12Hz architecture that is incompatible\n"
            f"with these export scripts. Use the 12Hz variant instead:\n"
            f"  1. Download: python download_models.py\n"
            f"     (this downloads {corrected})\n"
            f"  2. Then export: python export_lm.py --model-dir models/{_repo_to_local(corrected)}"
        )

    if repo_id in ALL_SUPPORTED_REPOS:
        local_name = _repo_to_local(repo_id)
        return (
            f"Download the model first, then use the local path:\n"
            f"  1. python download_models.py\n"
            f"  2. python export_lm.py --model-dir models/{local_name}"
        )

    return (
        f"Download a supported model first:\n"
        f"  python download_models.py --model customvoice\n"
        f"Then run the export with the local path:\n"
        f"  python export_lm.py --model-dir models/Qwen3-TTS-0.6B-CustomVoice"
    )


def _repo_to_local(repo_id: str) -> str:
    """Convert a HuggingFace repo ID to the expected local directory name."""
    name = repo_id.split("/")[-1]
    # Remove the "12Hz-" prefix that appears in repo IDs but not local dirs
    name = name.replace("12Hz-", "")
    return name


def _get_supported_for_type(script_type: str) -> list[str]:
    """Return supported repo IDs for a given script type."""
    if script_type in ("vocoder", "speech_tokenizer"):
        return SUPPORTED_TOKENIZER_REPOS
    if script_type == "speaker_encoder":
        return SUPPORTED_BASE_REPOS
    return SUPPORTED_CUSTOMVOICE_REPOS + SUPPORTED_BASE_REPOS
