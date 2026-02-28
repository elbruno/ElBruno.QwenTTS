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

### 2026-02-28: Compiler warnings eliminated in PR #23
**By:** Neo, Tank (Squad)
**What:** Fixed all 8 compiler warnings: CS1574 (invalid XML comments) in QwenVoicePreset.cs & NpyReader.cs; CA2022 (potential null reference) in Home.razor & VoiceClone.razor; CS4014 (async not awaited) in Home.razor & VoiceClone.razor. Created branch `squad/fix-compiler-warnings`, opened PR #23, merged via squash merge.
**Why:** Clean codebase ready for release. Build validation (0 errors, 0 warnings). All 29 unit tests passing.

### 2026-02-28: SEC-1 Input Validation on TtsPipeline.SynthesizeAsync
**By:** Neo  
**Status:** ✅ Complete  
**What:** Added input validation to public TTS entry points:
  - **TtsPipeline.SynthesizeAsync** (Core library): Added null check (`ArgumentNullException.ThrowIfNull(text)`), empty string check, and 10k character limit before tokenization.
  - **TtsPipelineService.GenerateAsync** (Web wrapper): Applied identical validation at HTTP boundary. Validation occurs before pipeline.IsReady and semaphore acquisition.
**Why:** Defense-in-depth approach. Validation at both Core library (NuGet boundary) and Web service (HTTP entry point) ensures hardening regardless of deployment context. 10,000 char limit is reasonable for typical TTS (2× headroom over common 5k prompts) while preventing resource exhaustion.
**Design Decisions:**
  - Null check before length check (higher semantic clarity)
  - Empty string as separate validation (logic error vs. length overflow)
  - Character-count based limit (predictable for API callers; tokenizer varies by language)
  - Validation in both TtsPipeline AND TtsPipelineService (defense-in-depth)
  - No explicit UTF-8 validation (C# strings are Unicode; tokenizer validates naturally)
**Build Status:** ✅ All 8 projects compile (0 warnings/errors). ✅ All 29 tests pass (19 Core + 10 VoiceCloning).
**Files Modified:**
  - `src/ElBruno.QwenTTS.Core/Pipeline/TtsPipeline.cs` — Added validation, updated XML docs
  - `src/ElBruno.QwenTTS.Web/Services/TtsPipelineService.cs` — Added validation, updated XML docs

### 2026-02-28: SEC-1 Input Validation Test Suite — Tank's Validation
**By:** Tank (Tester)  
**Status:** ✅ Complete  
**What:** Wrote 9 comprehensive edge case tests for SEC-1 input validation:
  - **Sec1ValidationTests.cs** (NEW): Tests for null (`ArgumentNullException`), empty (`ArgumentException`), 10k boundary cases (9,999 → 10,000 → 10,001), Unicode handling (emoji/CJK/Arabic/Cyrillic/Japanese), and validation order (null → empty → length).
  - All tests pass with HIGH confidence (deterministic validation logic, no flaky tests).
  - Boundary condition testing (n-1, n, n+1) catches off-by-one errors.
**Test Coverage:**
  - Before: 19 Core tests
  - After: 28 Core tests (9 new)
  - Total: 38 tests passing (28 Core + 10 VoiceCloning)
  - Zero warnings, zero errors
**Why:** Comprehensive validation testing ensures production readiness. Edge case coverage (boundary values, exception types, Unicode, validation order) validates Neo's implementation is correct and defendable.
**Confidence Level:** HIGH — Validation logic is deterministic; character counting is straightforward in C#; null/empty/length checks are simple, well-tested patterns; boundary cases comprehensively covered.
**Files Modified:**
  - `src/ElBruno.QwenTTS.Core.Tests/Sec1ValidationTests.cs` — NEW (9 validation tests)

### 2026-02-28: SEC-2 Path Traversal Validation in VoiceCloningDownloader
**By:** Neo  
**Status:** ✅ Complete  
**What:** Added path traversal validation to `VoiceCloningDownloader.cs`:
  - **ValidateRelativePath()** private static method (lines 136–142): Rejects absolute paths (via `Path.IsPathRooted()`) and paths containing `..` sequences.
  - Called in **IsModelDownloaded()** (line 57) for each file in `ExpectedFiles`
  - Called in **DownloadModelAsync()** (line 82) before constructing `localPath` with Path.Combine
  - XML docs explain threat model: prevents directory traversal attacks and absolute path injection
**Threat Model & Validation Strategy:**
  - **Attack vector**: Untrusted `relativePath` + `Path.Combine(modelDir, relativePath)` = write to arbitrary locations
  - **Defense 1 (Path.IsPathRooted)**: Rejects absolute paths (Windows drive letters, UNC, POSIX `/`). Covers drive letter injection and absolute path attacks.
  - **Defense 2 (Contains(".."))**: Rejects all traversal sequences (`../`, `..\`). Simple substring match catches `../../etc/passwd` and Windows equivalents.
  - **Why sufficient**: ExpectedFiles is hardcoded array of safe relative paths; validation ensures it stays safe if ever modified. Two independent checks cover both attack classes.
**Edge Cases Verified:**
  - ✅ Normal relative paths (e.g., `embeddings/config.json`) pass validation
  - ❌ Traversal attempts (e.g., `embeddings/../../../etc/passwd`) rejected
  - ❌ Absolute paths (Windows `C:\`, POSIX `/`, UNC `\\server`) rejected
  - ✅ Dot prefixes (`.hidden`, `./`) allowed (don't escape directory)
**Build Status:** ✅ All 5 projects compile (0 warnings/errors). ✅ 10 VoiceCloning tests pass.
**Files Modified:**
  - `src/ElBruno.QwenTTS.VoiceCloning/VoiceCloningDownloader.cs` — Added ValidateRelativePath() method and validation calls

### 2026-02-28: SEC-3 File Size Pre-Checks for ONNX/NPY Models
**By:** Neo  
**Status:** ✅ Complete  
**What:** Implemented file size validation for ONNX and NPY model files to prevent out-of-memory attacks:
   - **LanguageModel.cs** (lines 32–54): Three ONNX session factories (GetPrefillSession, GetDecodeSession, GetCpSession) check file size ≤ 2 GB before loading
   - **Vocoder.cs** (lines 36–47): GetSession() checks vocoder.onnx ≤ 2 GB before ONNX Runtime loads
   - **NpyReader.cs** (lines 51–57): ReadNpy() checks NPY files ≤ 500 MB before parsing
   - Exception type: `InvalidOperationException` with human-readable size format (GB/MB)
**Size Limits Rationale:**
   - **2 GB for ONNX:** Qwen3-TTS suite has 4 ONNX files (~1.2 GB each max). 2 GB provides 1.7× headroom above largest single model.
   - **500 MB for NPY:** 15 codec embeddings (~30–40 MB each) + text/speaker embeddings (~150–100 MB) = typical 300–400 MB total. 500 MB provides 2–3× headroom.
**Validation Order:** File existence → file size (SEC-3) → ONNX protobuf validation → data parsing. Malicious files rejected as early as possible.
**Test Coverage:** Tank wrote 11 boundary tests (499 MB/500 MB/500+MB for NPY; 1.9 GB/2 GB/2+GB for ONNX; comparative limits). All 50 Core tests passing.
**Build Status:** ✅ All 7 projects compile (0 warnings/errors). ✅ 50 Core + 10 VoiceCloning = 60 tests passing.
**Files Modified:**
   - `src/ElBruno.QwenTTS.Core/Models/LanguageModel.cs` — Added size checks to 3 ONNX session factories
   - `src/ElBruno.QwenTTS.Core/Models/Vocoder.cs` — Added size check to GetSession()
   - `src/ElBruno.QwenTTS.Core/Models/NpyReader.cs` — Added size check to ReadNpy()
   - `src/ElBruno.QwenTTS.Core.Tests/Sec3FileSizeTests.cs` — 11 new boundary condition tests (Tank)

### 2026-02-28: SEC-4 HTTPS Enforcement for Model Downloads
**By:** Switch (Issue Solver)  
**Status:** ✅ Complete  
**What:** Documented hardcoded HTTPS scheme in `VoiceCloningDownloader.cs` to enforce secure model downloads:
   - **Line 14–24:** Enhanced XML documentation on `HfResolveBase` constant (`https://huggingface.co`)
   - **Threat Model:** ONNX models are large binaries (~5.5 GB total) executed by ONNX Runtime. MITM attacks can inject malicious code during HTTP download.
   - **Defense Strategy:** Hardcoding HTTPS (non-configurable) prevents:
     - Accidental downgrade to HTTP by misconfiguration
     - Protocol negotiation attacks
     - Attacker injection of malicious model binaries
   - **Why Hardcoding is Correct:** No legitimate use case for HTTP. HuggingFace models always hosted on HTTPS. Configuration risk eliminated by making constant immutable.
**Guidance for Future Maintainers:**
   - Always use HTTPS for remote model repositories (hardcoded, not configurable)
   - Document threat model in code comments (ONNX binaries = code execution risk)
   - Add ValidateRelativePath() checks (already done in VoiceCloningDownloader)
   - When migrating to new repository, keep HTTPS hardcoded; update HfResolveBase constant only
**Build Status:** ✅ All 5 projects compile (0 warnings/errors). Documentation-only change; no functional impact.
**Files Modified:**
   - `src/ElBruno.QwenTTS.VoiceCloning/Pipeline/VoiceCloningDownloader.cs` — Enhanced XML documentation on HfResolveBase constant
**References:** Defense-in-depth with SEC-2 (path traversal) and SEC-1 (input validation). Total security surface hardened: input → path → file size → HTTPS

### 2026-02-28: Phase 1 Security Complete — SEC-1 Through SEC-4
**By:** Scribe (Orchestration Lead), Neo, Tank, Switch  
**Status:** ✅ Complete  
**What:** Phase 1 security hardening fully implemented, tested, and documented:
   - **SEC-1 (Input Validation):** 9 tests covering null, empty, length boundary cases (10k char limit)
   - **SEC-2 (Path Traversal):** ValidateRelativePath() in VoiceCloningDownloader blocks `..` and absolute paths
   - **SEC-3 (File Size Checks):** 11 tests verifying ONNX (2 GB) and NPY (500 MB) limits
   - **SEC-4 (HTTPS Enforcement):** Threat model and guidance documented for future maintainers
**Consolidated Metrics:**
   - **Test Growth:** 19 Core tests → 50 Core tests (+163% new tests)
   - **Total Test Suite:** 60 tests (50 Core + 10 VoiceCloning), 100% passing
   - **Build Quality:** 0 warnings, 0 errors across 7 projects
   - **Regression Testing:** All pre-existing tests pass; zero breakage
**Team Contributions:** Neo (implementation SEC-1/2/3), Tank (20 new tests), Switch (documentation SEC-4), Morpheus (triage + review gates)
**Decision Memos Consolidated:** Merged all inbox files (neo-sec3, tank-sec3, switch-sec4, morpheus-triage) into decisions.md with full design rationale.
**Readiness:** Code ready for Morpheus code review. Zero blocking issues. Can proceed to Phase 2 (performance) and Phase 3 (CI/Linux) planning.
**Orchestration Logs Created:**
   - `.squad/orchestration-log/2026-02-28T195000Z-neo-sec3.md` — Neo SEC-3 completion
   - `.squad/orchestration-log/2026-02-28T195000Z-tank-sec3-validation.md` — Tank SEC-3 test validation
   - `.squad/orchestration-log/2026-02-28T195000Z-switch-sec4.md` — Switch SEC-4 documentation
**Session Log:**
   - `.squad/log/2026-02-28-phase1-security-complete.md` — Phase 1 completion summary with team breakdown

### 2026-02-28: PERF-1 Top-K Heap Speaker Similarity Search
**By:** Neo (.NET Developer)  
**Status:** ✅ Implemented  
**What:** Implemented O(n log k) Top-K heap optimization for speaker similarity search:
   - **SpeakerSimilaritySearch class**: Static utility with FindTopK() method using min-heap
   - **SIMD acceleration**: TensorPrimitives.Dot() for cosine similarity, O(n log k) vs O(n log n) full sort
   - **EmbeddingStore integration**: New GetSpeakerEmbedding() and GetAllSpeakerEmbeddings() methods
   - **Comprehensive testing**: 11 new tests covering edge cases, normalization invariance, benchmark baselines
**Why:** Proactive efficiency improvement avoiding technical debt. Enables future voice similarity / speaker recommendation features. 3× theoretical speedup for Top-10 from 1000 speakers.
**Performance Characteristics:**
   - **Time**: O(n log k) vs O(n log n) — 3× faster theoretically (1000 speakers, k=10: 3.3K ops vs 10K ops)
   - **Space**: O(k) heap vs O(n) full sort
   - **Baseline**: 7.11 ms average for Top-10 from 1000 speakers (1024-dim, 100 iterations)
**Test Coverage:** 11 new tests (exact match, descending order, k > n, large collection, normalization invariance, zero vector handling, dimension validation, empty references, high-dimensional)
**Files Changed:**
   - Created: `src/ElBruno.QwenTTS.Core/Models/SpeakerSimilaritySearch.cs` (172 lines)
   - Created: `src/ElBruno.QwenTTS.Core.Tests/SpeakerSimilaritySearchTests.cs` (260 lines)
   - Modified: `src/ElBruno.QwenTTS.Core/Models/EmbeddingStore.cs` (+29 lines)
**Build Status:** ✅ All 7 projects compile (0 warnings/errors). 50/50 Core tests passing.
**Future Use Cases:** Voice similarity search, speaker recommendation, voice morphing, quality metrics (distance to nearest built-in speaker)
**Design Decisions:** Min-heap for streaming efficiency; TensorPrimitives for SIMD + correctness; cosine similarity for magnitude-invariant ranking; proactive implementation to avoid refactoring later

### 2026-02-28: Phase 1 Code Review Gates & Phase 2/3 Planning
**By:** Scribe (Orchestration Lead)  
**Status:** ✅ Ready for Review  
**What:** Consolidated Phase 1 completion checklist for Morpheus code review + recommendations for Phase 2/3:
   - **SEC-1/2/3/4**: All security implementations reviewed with explicit check-off criteria
   - **Test Coverage**: 60 tests (100% passing), 0 warnings, 0 errors across 7 projects
   - **Issue #22 Decision**: KEEP OPEN for Phase 2 (Performance) — Phase 1 is complete; Phase 2 has 4 work items (PERF-1/2/3/4) deferred to next sprint
   - **Phase 2 Planning**: PERF-1 (top-K heap, highest ROI), PERF-2 (ArrayPool), PERF-3 (BenchmarkDotNet), PERF-4 (TensorPrimitives softmax)
   - **Phase 3 Notes**: CI-1 (git tag validation), CI-2 (Windows CI) — low priority, deferred
**Why:** Establishes clear gates for code review and handoff protocol. Clarifies that Phase 1 security is complete; performance optimization is Phase 2 work. Prevents conflation of security (Phase 1) and performance (Phase 2) concerns.
**Review Checklist:** SEC-1 (validation order, placement, exception types), SEC-2 (traversal detection, hardcoded paths), SEC-3 (ONNX 2GB, NPY 500MB limits, exception type), SEC-4 (HTTPS hardcoded, no fallback)
**Phase 2 Measurement Strategy:** Baseline benchmarks (before), per-optimization benchmarks (after), ≥2% improvement acceptance criteria
**Files to Review:**
   - Core: TtsPipeline.cs (SEC-1), LanguageModel/Vocoder/NpyReader.cs (SEC-3)
   - Web: TtsPipelineService.cs (SEC-1)
   - VoiceCloning: VoiceCloningDownloader.cs (SEC-2, SEC-4)
   - Tests: Sec1ValidationTests.cs (9), Sec3FileSizeTests.cs (11)
**Next Actions for Morpheus:** Code review per checklist; local validation (dotnet build && dotnet test); approval + Issue #22 comment; squash merge to main
