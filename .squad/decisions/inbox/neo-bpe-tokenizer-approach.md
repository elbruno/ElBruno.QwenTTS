### BPE Tokenizer: Microsoft.ML.Tokenizers BpeTokenizer with ByteLevel

**By:** Neo
**When:** 2026-02-21
**What:** Used `BpeTokenizer.Create(BpeOptions)` from Microsoft.ML.Tokenizers 2.0.0 with `ByteLevel = true` and a `RegexPreTokenizer` carrying the GPT-2 pre-tokenization regex plus all Qwen3-TTS special tokens. This loads vocab.json + merges.txt directly via the `BpeOptions(vocabFile, mergesFile)` constructor — no manual parsing needed.
**Why:** BpeTokenizer with BpeOptions gives full control over byte-level encoding, pre-tokenization regex, and special token registration. The alternative `CodeGenTokenizer` lacked special-token passthrough and the simpler `BpeTokenizer.Create(file, file)` overload didn't expose the ByteLevel flag. This approach should produce token IDs matching the Python HuggingFace tokenizer once we validate against `validation_cases.json`.
**Risk:** The GPT-2 regex pattern may differ slightly from the actual Qwen2 pre-tokenization regex in their `tokenizer.json`. If validation fails, we may need to extract the exact regex from the Python tokenizer artifacts and adjust.
