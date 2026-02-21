# Qwen3-TTS Tokenizer & Prompt Format Reference

> **For:** Neo (C# developer) — everything needed to reproduce Qwen3-TTS tokenization and prompt construction in C#.
>
> **Model:** `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
>
> **Analyzed by:** Trinity (ML Engineer), from upstream source at [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

---

## 1. Tokenizer Type

The text tokenizer is a **Qwen2Tokenizer** — a GPT-2 style **Byte-Pair Encoding (BPE)** tokenizer. The underlying implementation uses the HuggingFace `tokenizers` library (Rust-based fast tokenizer).

**Key files for C# re-implementation:**

| File | Purpose |
|------|---------|
| `vocab.json` | Maps token strings → integer IDs (151,936 base entries) |
| `merges.txt` | BPE merge rules, applied in order (pair → merged token) |
| `tokenizer_config.json` | Configuration: special tokens, EOS/BOS, etc. |
| `special_tokens_map.json` | Maps roles (eos_token, pad_token, etc.) to token strings |

These are extracted by `extract_tokenizer.py` into `python/tokenizer_artifacts/`.

**Tokenization algorithm:** Standard BPE with byte-level fallback. Text is first UTF-8 encoded, then bytes are mapped to the base vocabulary via the pre-tokenization regex, and then BPE merges are applied greedily in the order listed in `merges.txt`.

---

## 2. Special Tokens & Their IDs

### Core Special Tokens (Text Tokenizer)

| Token | ID | Role |
|-------|------|------|
| `<\|endoftext\|>` | 151643 | Pad token |
| `<\|im_start\|>` | 151644 | Chat turn start marker |
| `<\|im_end\|>` | 151645 | Chat turn end marker / EOS |

> **Note:** `bos_token` is `null` (not used). The EOS token is `<|im_end|>` (ID 151645).

### TTS-Specific Special Tokens (Text Tokenizer)

| Token | ID | Role |
|-------|------|------|
| `<tts_pad>` | 151671 | TTS pad embedding (used in codec prefix construction) |
| `<tts_text_bos>` | 151672 | TTS text beginning-of-sequence |
| `<tts_text_eod>` | 151673 | TTS text end-of-document |
| `<tts_text_bos_single>` | 151674 | TTS single-utterance BOS |
| `<\|audio_start\|>` | 151669 | Audio region start |
| `<\|audio_end\|>` | 151670 | Audio region end |
| `<\|audio_pad\|>` | 151675 | Audio padding |

### Codec Control Tokens (Talker LM Vocabulary, IDs in range 0–3071)

These are **NOT** text token IDs. They live in the Talker LM's codec vocabulary (vocab_size=3072) and are used as embedding indices for the Talker's `nn.Embedding`:

| Name | Codec ID | Role |
|------|----------|------|
| `codec_pad` | 2148 | Padding in codec embedding space |
| `codec_bos` | 2149 | Beginning-of-speech (starts audio generation) |
| `codec_eos` | 2150 | End-of-speech (stops generation when predicted as group-0) |
| `codec_think` | 2154 | "Think" mode tag — used when language is explicitly specified |
| `codec_nothink` | 2155 | "No-think" mode tag — used when language is "auto" |
| `codec_think_bos` | 2156 | Think section start delimiter |
| `codec_think_eos` | 2157 | Think section end delimiter |

### Language IDs (Codec Vocabulary)

These are codec token IDs used inside the "think" section to specify the target language:

| Language | Codec ID |
|----------|----------|
| English | 2050 |
| German | 2053 |
| Spanish | 2054 |
| Chinese | 2055 |
| Japanese | 2058 |
| French | 2061 |
| Korean | 2064 |
| Russian | 2069 |
| Italian | 2070 |
| Portuguese | 2071 |
| Beijing dialect | 2074 |
| Sichuan dialect | 2062 |

### Speaker IDs (Codec Vocabulary)

These are codec token IDs used to embed the speaker identity:

| Speaker | Codec ID | Dialect |
|---------|----------|---------|
| Serena | 3066 | — |
| Vivian | 3065 | — |
| Uncle_Fu | 3010 | — |
| Ryan | 3061 | — |
| Aiden | 2861 | — |
| Ono_Anna | 2873 | — |
| Sohee | 2864 | — |
| Eric | 2875 | Sichuan dialect |
| Dylan | 2878 | Beijing dialect |

> **Dialect auto-selection:** When language is "chinese" or "auto" and the speaker has a dialect mapping (Eric → Sichuan, Dylan → Beijing), the language_id is automatically replaced with the dialect's codec ID.

---

## 3. Prompt Format — How Text Is Prepared

### 3a. Text Wrapping (Chat Template)

The `generate_custom_voice()` method wraps the user's text using the `_build_assistant_text()` helper:

```python
def _build_assistant_text(self, text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
```

**Example:** For input text `"Hello, world!"`, the tokenizer receives:

```
<|im_start|>assistant\nHello, world!<|im_end|>\n<|im_start|>assistant\n
```

This produces token IDs like: `[151644, 77091, 198, ..., 151645, 198, 151644, 77091, 198]`

Where:
- `151644` = `<|im_start|>`
- `77091` = `assistant`
- `198` = `\n`
- `...` = BPE tokens for the actual text
- `151645` = `<|im_end|>`

### 3b. Instruct Wrapping

When an instruction is provided, it's wrapped as a user turn:

```python
def _build_instruct_text(self, instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"
```

**Example:** `"Speak slowly and clearly."` becomes:

```
<|im_start|>user\nSpeak slowly and clearly.<|im_end|>\n
```

> **Note:** For the 0.6B model (`tts_model_size == "0b6"`), instruct is **forced to None** and not used. Only larger models support instruction control.

### 3c. Reference Text Wrapping (Voice Clone only — not used in CustomVoice)

```python
def _build_ref_text(self, text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"
```

---

## 4. Full Embedding Construction (What C# Must Reproduce)

After tokenization, the model builds a combined embedding sequence for the Talker LM. Here is the exact layout:

### Step 1: Tokenize the wrapped text

```
text_tokens = tokenize("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
```

### Step 2: Split the token sequence

The code splits the tokenized sequence into parts:
- `input_id[:, :3]` → First 3 tokens: `<|im_start|>`, `assistant`, `\n` (the role prefix)
- `input_id[:, 3:-5]` → The actual text content tokens
- `input_id[:, -5:]` → The suffix: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`

### Step 3: Build the codec prefix embedding

The codec prefix is built from Talker LM embeddings (NOT text embeddings):

**When language = "auto" (no explicit language):**
```
codec_prefix = [codec_nothink(2155), codec_think_bos(2156), codec_think_eos(2157)]
```

**When language is explicitly specified (e.g., "english"):**
```
codec_prefix = [codec_think(2154), codec_think_bos(2156), language_id(2050), codec_think_eos(2157)]
```

Then the suffix tokens are appended:
```
codec_suffix = [codec_pad(2148), codec_bos(2149)]
```

**With speaker:**
```
codec_embedding = [codec_prefix..., speaker_embed(spk_id), codec_pad(2148), codec_bos(2149)]
```

**Without speaker:**
```
codec_embedding = [codec_prefix..., codec_pad(2148), codec_bos(2149)]
```

### Step 4: Combine text + codec embeddings

The model combines text and codec embeddings element-wise:

```
role_embed = text_projection(text_embedding(input_id[:3]))  # <|im_start|> assistant \n

# Pad embeddings overlaid with codec embeddings
pad_embeds = tts_pad_embed × (len(codec_embedding) - 2)  # repeated
talker_prefix = concat(pad_embeds, tts_bos_embed) + codec_embedding[:-1]

# Full prefill
talker_input = concat(role_embed, talker_prefix, text_content + codec_bos, tts_eos + codec_pad)
```

### Step 5: Non-streaming mode (default)

In non-streaming mode (`non_streaming_mode=True`, the default), ALL text tokens are included in the prefill:

```
prefill = [role_embed, codec_prefix_embed, text_content_embed + codec_pad_embed, tts_eos_embed + codec_pad_embed, tts_pad_embed + codec_bos_embed]
```

The `trailing_text_hidden` is set to `tts_pad_embed` (a constant), meaning no additional text is streamed during generation.

### Embedding Combination Summary

Each position in the prefill sequence is the **sum** of:
1. A **text-side embedding** (projected through `text_projection`: 2048d → 1024d)
2. A **codec-side embedding** (from Talker's `nn.Embedding`: 1024d)

This dual-embedding structure is critical for C# to reproduce correctly.

---

## 5. Instruct Embedding Placement

When instruct text is provided, it is:
1. Tokenized: `tokenize("<|im_start|>user\n{instruct}<|im_end|>\n")`
2. Embedded via `text_embedding()` then `text_projection()`
3. **Prepended** to the talker input (before the main text/codec sequence)

The instruct embedding appears at the beginning of the full sequence:
```
full_input = [instruct_embed, role_embed, codec_prefix_embed, text+codec_embed...]
```

---

## 6. Key Implementation Notes for C#

### What C# must implement:

1. **BPE Tokenizer** — Use `vocab.json` + `merges.txt` to reproduce `tokenizer.encode()`. The validation cases in `validation_cases.json` verify correctness.

2. **Chat template wrapping** — Apply the exact format from §3a before tokenizing.

3. **Text embedding + projection** — Table lookup (151,936 × 2048) → Linear projection (2048 → 1024). These weights come from the Talker model's `text_embedding` and `text_projection` layers.

4. **Codec embedding** — Table lookup in the Talker's main embedding (3072 × 1024) for codec control tokens, language IDs, and speaker IDs.

5. **Element-wise sum** — Combine text and codec embeddings as described in §4.

### What C# does NOT need:

- The tokenizer does **not** use BOS tokens (`add_bos_token: false`).
- No sentencepiece — this is pure BPE with byte-level fallback.
- No audio feature extraction for CustomVoice (that's only for the Base voice-clone model).

### Token ID 77091

The token `assistant` (as a single token) has ID **77091**. This is referenced in the model config as `assistant_token_id` and appears in the chat template prefix.

---

## 7. Validation Strategy

Run `extract_tokenizer.py` to generate `validation_cases.json`. Each entry contains:
```json
{
  "description": "simple_english",
  "text": "Hello, world!",
  "token_ids": [9707, 11, 1879, 0],
  "num_tokens": 4
}
```

The C# tokenizer should produce **exactly** the same `token_ids` for each `text` input. The cases cover:
- English, Chinese, mixed language
- Special characters, punctuation, numbers
- Whitespace edge cases
- Full chat-template-wrapped strings (what the model actually sees)
- Instruct format strings

---

## 8. Quick Reference: Full CustomVoice Prompt Assembly

```
Given: text="Hello!", speaker="Ryan", language="english"

1. Wrap text:
   wrapped = "<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n"

2. Tokenize wrapped text:
   tokens = [151644, 77091, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
            [im_start, assistant, \n, Hello, !, im_end, \n, im_start, assistant, \n]

3. Split tokens:
   role_tokens   = tokens[:3]     = [151644, 77091, 198]
   text_tokens   = tokens[3:-5]   = [9707, 0]  (actual text content)
   suffix_tokens = tokens[-5:]    = [151645, 198, 151644, 77091, 198]

4. Build codec prefix (language="english" → use think mode):
   codec_prefix = [2154, 2156, 2050, 2157]  (think, think_bos, english, think_eos)

5. Build codec suffix with speaker:
   speaker_id = 3061  (Ryan)
   codec_full = [2154, 2156, 2050, 2157, speaker_embed(3061), 2148, 2149]
                [think, think_bos, eng, think_eos, speaker, pad, bos]

6. Build prefill embedding (non-streaming):
   pos 0-2:   text_proj(text_embed(role_tokens))
   pos 3:     tts_pad_embed + codec_embed(2154)    # think
   pos 4:     tts_pad_embed + codec_embed(2156)    # think_bos
   pos 5:     tts_pad_embed + codec_embed(2050)    # english
   pos 6:     tts_pad_embed + codec_embed(2157)    # think_eos
   pos 7:     tts_pad_embed + speaker_embed(3061)  # speaker
   pos 8:     tts_bos_embed + codec_embed(2148)    # pad
   pos 9:     text_proj(text_embed("Hello")) + codec_embed(2148)  # text + pad
   pos 10:    text_proj(text_embed("!")) + codec_embed(2148)      # text + pad
   pos 11:    tts_eos_embed + codec_embed(2148)                   # eos + pad
   pos 12:    tts_pad_embed + codec_embed(2149)                   # pad + bos

   → Total prefill length = 3 (role) + len(codec_full)-1 + len(text_content)+1 + 1

7. Trailing text hidden = tts_pad_embed (constant, non-streaming mode)

8. Feed prefill → Talker LM → autoregressive generation → codec tokens
```

---

## 9. Config Token IDs (from model config.json)

```json
{
  "im_start_token_id": 151644,
  "im_end_token_id": 151645,
  "assistant_token_id": 77091,
  "tts_bos_token_id": 151672,
  "tts_eos_token_id": 151673,
  "tts_pad_token_id": 151671
}
```

These are the text-tokenizer IDs that get looked up in `text_embedding` (151,936 × 2048) and then projected to 1024d via `text_projection`. They are used to create the special TTS embeddings (`tts_bos_embed`, `tts_eos_embed`, `tts_pad_embed`) that form the backbone of the prefill sequence.
