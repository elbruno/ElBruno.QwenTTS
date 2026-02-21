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
