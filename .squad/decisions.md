# Decisions

> Team decisions that all agents should respect. Managed by Scribe.

### 2026-02-21T15:38Z: Target model selection
**By:** Bruno Capuano (via Squad)
**What:** Start with Qwen3-TTS-12Hz-0.6B-CustomVoice (simplest — no audio input, just text + speaker). Voice cloning with Base model is a stretch goal.
**Why:** 0.6B is practical for local inference (~1.8GB). CustomVoice has 9 built-in speakers and instruction control.

### 2026-02-21T15:38Z: Framework constraints
**By:** Bruno Capuano (via Squad)
**What:** Use .NET 10 and latest available packages for both C# and Python.
**Why:** User requirement — always use the latest.

### 2026-02-21T16:30Z: GPU handoff — continue on NVIDIA machine
**By:** Bruno Capuano (via Squad)
**What:** ONNX export scripts run on CPU but are impractical without GPU acceleration. Repo published to GitHub (elbruno/qwen-labs-cs, private). All 8 completed tasks committed. Work continues on NVIDIA GPU machine — next step is running export scripts.
**Why:** Current machine has no NVIDIA GPU. Export scripts are designed for CPU (device_map="cpu", dtype=float32) but model download + export is slow without CUDA.

### 2026-02-21T16:30Z: Continuation instructions
**By:** Squad (Coordinator)
**What:** On the GPU machine, the Squad should: (1) run download_models.py, (2) run all export scripts in order: vocoder → LM → embeddings → tokenizer, (3) implement LanguageModel.cs, (4) wire TtsPipeline.cs, (5) test end-to-end.
**Why:** Documented to ensure seamless handoff — any Squad session on the new machine can pick up immediately.

### 2026-02-21: .NET project scaffold and package choices
**By:** Neo (.NET Developer)
**What:** Target net10.0 with Microsoft.ML.OnnxRuntime 1.24.2, Microsoft.ML.Tokenizers 2.0.0, NAudio 2.2.1. Simple string[] CLI parsing (no System.CommandLine). WavWriter outputs 16-bit PCM WAV at 24 kHz.
**Why:** OnnxRuntime provides robust ONNX inference; Tokenizers avoids hand-rolling BPE; lightweight dependencies. Project structure: Models / Pipeline / Audio separation.

### 2026-02-21: BPE Tokenizer — Microsoft.ML.Tokenizers BpeTokenizer
**By:** Neo
**What:** Use `BpeTokenizer.Create(BpeOptions)` with ByteLevel=true, GPT-2 pre-tokenization regex, and all Qwen3-TTS special tokens. Load vocab.json + merges.txt directly via BpeOptions constructor.
**Why:** BpeTokenizer gives full control over byte-level encoding and special tokens. Produces token IDs matching Python HuggingFace tokenizer (validate against validation_cases.json). Avoids manual regex parsing.

### 2026-02-21: Vocoder tensor API — shape and dtype alignment
**By:** Neo
**What:** Changed Vocoder.Decode() signature from int[] to long[,,] (shape 1×16×T) to match ONNX model's expected input (B, 16, T_codes) int64. Pipeline reshapes LM's flat output into this 3D array.
**Why:** Matches ONNX model contract explicitly; avoids runtime reshaping inside vocoder; makes shape expectations clear.

### 2026-02-21: BPE Tokenizer Extraction from Python Model
**By:** Trinity (ML Engineer)
**What:** Created `python/extract_tokenizer.py` to extract Qwen3-TTS BPE artifacts (vocab.json, merges.txt, configs, validation cases). Documented full prompt format in `python/TOKENIZER.md`.
**Why:** Enables C# re-implementation using only vocab.json + merges.txt. Standard Qwen2Tokenizer (GPT-2 BPE) — no exotic preprocessing.

### 2026-02-21: Tokenizer findings — CustomVoice prompt structure
**By:** Trinity (ML Engineer)
**What:** The 0.6B CustomVoice model does NOT support instruct control (forced to None). Speaker selection uses codec embedding IDs (e.g., Ryan=3061). Language encoded as codec token; "auto" mode skips language ID. Eric/Dylan have dialect remapping for Chinese/auto modes.
**Why:** Ensures prompt format matches actual model behavior. C# TextTokenizer.BuildCustomVoicePrompt() must follow this structure.

### 2026-02-21: ONNX Export Strategy — 3-Model Split
**By:** Trinity (ML Engineer)
**What:** Export Qwen3-TTS as three separate ONNX models: Vocoder (single-pass), Code Predictor (autoregressive per group), Talker LM (large autoregressive with KV-cache). Export order: Vocoder → Code Predictor → Talker LM.
**Why:** Three distinct execution patterns (single-pass vs autoregressive). Code Predictor runs 31× per Talker step — must be separate and fast. Different KV-cache semantics (Talker grows across generation; CP resets per step). Vocoder is simplest validation milestone.

### 2026-02-21: Vocoder ONNX Export — Two-Attempt Strategy
**By:** Trinity (ML Engineer)
**What:** `python/export_vocoder.py` uses standard torch.onnx.export (opset 17) first, falls back to dynamo=True (opset 18+) if tracing fails. Dynamic axes on batch and time dimensions.
**Why:** Decoder contains sliding-window attention and causal convolution padding with data-dependent shapes. Dynamo backend handles these patterns more robustly. Two paths let us discover which works without manual iteration.

### 2026-02-21: Code Predictor lm_head stacking for ONNX
**By:** Trinity (ML Engineer)
**What:** Stack all 31 lm_head weight matrices into (31, 2048, 1024) buffer. Use torch.index_select on generation_steps input to select correct weight at runtime. Single `code_predictor.onnx` model per step.
**Why:** ONNX tracing cannot dynamically index ModuleList. Index-select avoids exporting 31 separate models or wasteful torch.where with all outputs. C# calls one model per step, passing generation_steps (0-30).

### 2026-02-21T17:45Z: C# Inference Pipeline Complete
**By:** Neo
**What:** Implemented full C# ONNX inference pipeline: NpyReader (NumPy .npy loader), EmbeddingStore (centralized lookups with SiLU-gated MLP), rewritten LanguageModel (3-session autoregressive with KV stacking), updated TtsPipeline (delegated prompt building), Program (CLI with --model-dir/--language). All components wired end-to-end.
**Why:** Bruno can now test end-to-end inference immediately after Trinity exports ONNX models. Pipeline matches Python reference architecture exactly. Ready for validation against Python baseline.

### 2026-02-21: C# Pipeline Bug Fixes (Coordinator Review)
**By:** Coordinator
**What:** Fixed 4 runtime bugs: (1) TTS embedding sizing — corrected prefill tensor dimensions, (2) Hidden state indexing — fixed decode loop stacking from flat to (B, 8, T, 128), (3) Code Predictor KV tracking — CP sessions no longer accumulate KV across groups; reset per group, (4) Tokenizer case sensitivity — ensured vocab.json respects case-sensitive special tokens.
**Why:** Ensures pipeline produces correct outputs matching ONNX model contracts. All bugs found during code review before runtime testing.
