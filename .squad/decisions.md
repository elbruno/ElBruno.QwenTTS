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
