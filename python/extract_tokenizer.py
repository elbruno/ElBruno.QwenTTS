#!/usr/bin/env python3
"""
Extract BPE tokenizer artifacts from Qwen3-TTS-12Hz-0.6B-CustomVoice for C# consumption.

Saves:
  - tokenizer_artifacts/vocab.json          (BPE vocabulary: token_str -> token_id)
  - tokenizer_artifacts/merges.txt          (BPE merge rules)
  - tokenizer_artifacts/tokenizer_config.json
  - tokenizer_artifacts/special_tokens_map.json
  - tokenizer_artifacts/validation_cases.json  (test cases for C# validation)

Requires: transformers, tokenizers
Run:  python extract_tokenizer.py
"""

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_DIR = Path(__file__).parent / "tokenizer_artifacts"


def extract_tokenizer_files(tokenizer, output_dir: Path) -> None:
    """Save the raw tokenizer files that a C# BPE implementation needs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the tokenizer — this writes vocab.json, merges.txt,
    # tokenizer_config.json, special_tokens_map.json, etc.
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved tokenizer files to {output_dir}")

    # List what was written
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


def build_validation_cases(tokenizer) -> list[dict]:
    """
    Build validation test cases for the C# tokenizer to verify against.

    Each case is {"text": str, "token_ids": list[int], "description": str}.
    The text here is the RAW text fed to the tokenizer (pre-chat-template).
    The C# side should produce identical token_ids for the same input.
    """
    cases = []

    def add(description: str, text: str) -> None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({
            "description": description,
            "text": text,
            "token_ids": ids,
            "num_tokens": len(ids),
        })

    # --- English ---
    add("simple_english", "Hello, world!")
    add("english_sentence", "The quick brown fox jumps over the lazy dog.")
    add("english_tts_phrase", "Welcome to the Qwen text to speech system.")

    # --- Chinese ---
    add("simple_chinese", "你好世界")
    add("chinese_sentence", "今天天气真不错，我们一起出去走走吧。")

    # --- Mixed language ---
    add("mixed_en_zh", "Hello你好，Welcome欢迎!")
    add("mixed_with_numbers", "第3次世界大战在2045年")

    # --- Special characters & punctuation ---
    add("punctuation", "Hello! How are you? I'm fine, thanks.")
    add("special_chars", "Email: user@example.com — price: $19.99")
    add("unicode_symbols", "Temperature: 72°F → 22°C ✓")

    # --- Numbers ---
    add("pure_numbers", "1234567890")
    add("decimal_number", "3.14159265")

    # --- Whitespace edge cases ---
    add("single_space", " ")
    add("single_char", "A")
    add("newline", "\n")

    # --- Chat template format (what the model actually sees) ---
    # This is the format used by generate_custom_voice():
    #   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    assistant_text = "<|im_start|>assistant\nHello, world!<|im_end|>\n<|im_start|>assistant\n"
    add(
        "chat_template_english",
        assistant_text,
    )

    assistant_text_zh = "<|im_start|>assistant\n你好世界<|im_end|>\n<|im_start|>assistant\n"
    add(
        "chat_template_chinese",
        assistant_text_zh,
    )

    # Instruct format: <|im_start|>user\n{instruct}<|im_end|>\n
    instruct_text = "<|im_start|>user\nSpeak slowly and clearly.<|im_end|>\n"
    add(
        "instruct_format",
        instruct_text,
    )

    return cases


def save_special_tokens_summary(tokenizer, output_dir: Path) -> None:
    """Save a human-readable summary of all special tokens and their IDs."""
    summary = {
        "eos_token": {"token": str(tokenizer.eos_token), "id": tokenizer.eos_token_id},
        "pad_token": {"token": str(tokenizer.pad_token), "id": tokenizer.pad_token_id},
        "bos_token": {"token": str(tokenizer.bos_token), "id": tokenizer.bos_token_id},
    }

    # Collect all added tokens (special tokens beyond the base vocab)
    added_tokens = {}
    if hasattr(tokenizer, "added_tokens_encoder"):
        for token_str, token_id in sorted(
            tokenizer.added_tokens_encoder.items(), key=lambda x: x[1]
        ):
            added_tokens[token_str] = token_id

    summary["added_tokens"] = added_tokens
    summary["vocab_size"] = tokenizer.vocab_size

    out_path = output_dir / "special_tokens_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved special tokens summary to {out_path}")


def main() -> None:
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 1. Extract raw tokenizer files
    extract_tokenizer_files(tokenizer, OUTPUT_DIR)

    # 2. Save special tokens summary
    save_special_tokens_summary(tokenizer, OUTPUT_DIR)

    # 3. Generate and save validation cases
    cases = build_validation_cases(tokenizer)
    validation_path = OUTPUT_DIR / "validation_cases.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cases)} validation cases to {validation_path}")

    # Print a quick preview
    print("\n--- Validation Case Preview ---")
    for case in cases:
        desc = case["description"]
        text_preview = case["text"][:40].replace("\n", "\\n")
        n = case["num_tokens"]
        print(f"  {desc:30s} | {n:3d} tokens | {text_preview!r}")

    print("\nDone. All artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
