#!/usr/bin/env python3
"""
One-command environment setup for Qwen3-TTS ONNX + C# pipeline.

This script:
  1. Checks prerequisites (Python ≥ 3.10, .NET 10 SDK)
  2. Installs the huggingface_hub package (if missing)
  3. Downloads ONNX models from HuggingFace (~5.5 GB)
  4. Verifies all model files are present
  5. Restores .NET NuGet packages

Usage:
  python setup_environment.py                          # defaults
  python setup_environment.py --model-dir ./my-models  # custom output dir
  python setup_environment.py --skip-dotnet            # Python-only setup
"""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_REPO_ID = "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX"
DEFAULT_MODEL_DIR = os.path.join("python", "onnx_runtime")
CSPROJ_PATH = os.path.join("src", "QwenTTS", "QwenTTS.csproj")

EXPECTED_FILES = [
    "talker_prefill.onnx",
    "talker_prefill.onnx.data",
    "talker_decode.onnx",
    "talker_decode.onnx.data",
    "code_predictor.onnx",
    "vocoder.onnx",
    "vocoder.onnx.data",
    os.path.join("embeddings", "config.json"),
    os.path.join("embeddings", "talker_codec_embedding.npy"),
    os.path.join("embeddings", "text_embedding.npy"),
    os.path.join("embeddings", "text_projection_fc1_weight.npy"),
    os.path.join("embeddings", "text_projection_fc1_bias.npy"),
    os.path.join("embeddings", "text_projection_fc2_weight.npy"),
    os.path.join("embeddings", "text_projection_fc2_bias.npy"),
    os.path.join("embeddings", "codec_head_weight.npy"),
    os.path.join("embeddings", "speaker_ids.json"),
    os.path.join("tokenizer", "vocab.json"),
    os.path.join("tokenizer", "merges.txt"),
]

# CP codec embeddings (groups 0-14)
for i in range(15):
    EXPECTED_FILES.append(os.path.join("embeddings", f"cp_codec_embedding_{i}.npy"))


# ── Helpers ────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg):
    print(f"  {RED}✗ {msg}{RESET}")
    sys.exit(1)


def header(msg):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def run_cmd(cmd, check=True, capture=False):
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True
    )
    if check and result.returncode != 0:
        return None
    return result


# ── Step 1: Check prerequisites ───────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        info(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python 3.10+ required (found {v.major}.{v.minor}.{v.micro})")
    return False


def check_dotnet():
    result = run_cmd("dotnet --version", check=False, capture=True)
    if result and result.returncode == 0:
        version = result.stdout.strip()
        info(f".NET SDK {version}")
        return True
    warn(".NET SDK not found — install from https://dotnet.microsoft.com/download")
    return False


# ── Step 2: Install Python dependencies ───────────────────────────────────────

def ensure_huggingface_hub():
    try:
        importlib.import_module("huggingface_hub")
        info("huggingface_hub already installed")
    except ImportError:
        print(f"  ↓ Installing huggingface_hub...")
        result = run_cmd(
            f"{sys.executable} -m pip install --quiet huggingface_hub",
            check=False,
        )
        if result and result.returncode == 0:
            info("huggingface_hub installed")
        else:
            fail("Failed to install huggingface_hub. Run: pip install huggingface_hub")


# ── Step 3: Download ONNX models ──────────────────────────────────────────────

def download_models(repo_id, output_dir):
    from huggingface_hub import hf_hub_download, list_repo_files

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id)
    files = [f for f in files if f != "README.md" and not f.startswith(".")]

    total = len(files)
    downloaded = 0
    skipped = 0

    for idx, filename in enumerate(sorted(files), 1):
        local_path = output_path / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            skipped += 1
            print(f"  {DIM}[{idx}/{total}] ✓ {filename} (exists){RESET}")
            continue

        print(f"  [{idx}/{total}] ↓ {filename} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path),
        )
        downloaded += 1

    info(f"Download complete: {downloaded} new, {skipped} cached, {total} total")
    return output_path


# ── Step 4: Verify model files ────────────────────────────────────────────────

def verify_models(model_dir):
    model_path = Path(model_dir)
    missing = []
    for f in EXPECTED_FILES:
        if not (model_path / f).exists():
            missing.append(f)

    if missing:
        warn(f"Missing {len(missing)} file(s):")
        for f in missing[:5]:
            print(f"    - {f}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
        return False

    info(f"All {len(EXPECTED_FILES)} model files verified")
    return True


# ── Step 5: Restore .NET packages ─────────────────────────────────────────────

def restore_dotnet():
    if not Path(CSPROJ_PATH).exists():
        warn(f"C# project not found at {CSPROJ_PATH} — skipping NuGet restore")
        return

    print(f"  ↓ Restoring NuGet packages...")
    result = run_cmd(
        f"dotnet restore {CSPROJ_PATH}",
        check=False,
        capture=True,
    )
    if result and result.returncode == 0:
        info("NuGet packages restored")
    else:
        warn("NuGet restore failed — run manually: dotnet restore src/QwenTTS/QwenTTS.csproj")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Set up the Qwen3-TTS ONNX + C# environment"
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to download models to (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--skip-dotnet",
        action="store_true",
        help="Skip .NET SDK check and NuGet restore",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}Qwen3-TTS Environment Setup{RESET}")
    print(f"   Model repo: {args.repo_id}")
    print(f"   Model dir:  {args.model_dir}")

    # Step 1: Prerequisites
    header("Step 1 · Checking prerequisites")
    check_python()
    if not args.skip_dotnet:
        check_dotnet()

    # Step 2: Python dependencies
    header("Step 2 · Installing Python dependencies")
    ensure_huggingface_hub()

    # Step 3: Download models
    header("Step 3 · Downloading ONNX models (~5.5 GB)")
    download_models(args.repo_id, args.model_dir)

    # Step 4: Verify
    header("Step 4 · Verifying model files")
    verify_models(args.model_dir)

    # Step 5: .NET
    if not args.skip_dotnet:
        header("Step 5 · Restoring .NET packages")
        restore_dotnet()

    # Done
    header("Setup complete!")
    model_dir = args.model_dir.replace("\\", "/")
    print(f"""
  Run the C# app:

    dotnet run --project src/QwenTTS -- \\
      --model-dir {model_dir} \\
      --text "Hello, this is a test." \\
      --speaker ryan \\
      --language english \\
      --output hello.wav

  Available speakers:
    serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan
""")


if __name__ == "__main__":
    main()
