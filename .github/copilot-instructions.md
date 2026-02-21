# Copilot Instructions

This is a **Squad-managed** repository (Squad v0.5.2) — an AI team framework where a Coordinator orchestrates specialist agents.

## Application

This repo contains a **Qwen3-TTS ONNX Pipeline + C# .NET 10 Console App** for local text-to-speech. Pre-exported ONNX models are hosted on HuggingFace at `elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX`.

## Documentation Convention

- **Only `README.md` and `LICENSE` live at the repo root.** All other documentation goes in the `docs/` folder.
- Documentation files use kebab-case naming (e.g., `docs/getting-started.md`, `docs/architecture.md`).
- The main `README.md` should be concise with quick start commands and links to `docs/` for details.

## Repository Structure

- `docs/` — All documentation (prerequisites, getting started, architecture, exporting, troubleshooting, etc.)
- `src/QwenTTS/` — C# .NET 10 console application
- `python/` — ONNX export & download tools
- `setup_environment.py` — One-command setup script
- `.squad/` — **User-owned** team state (decisions, identity, ceremonies, skills, logs). Never overwritten by upgrades.
- `.squad-templates/` — **Squad-owned** templates. Overwritten on upgrade — do not customize here.
- `.github/agents/squad.agent.md` — Coordinator prompt (Squad-owned).
- `.github/workflows/` — Squad CI/CD workflows (triage, heartbeat, label sync, release pipeline).
- `.copilot/mcp-config.json` — MCP server configuration.

## Key Conventions

### File Ownership Model

Squad distinguishes **user-owned** files (`.squad/`) from **Squad-owned** files (`.squad-templates/`, `.github/agents/squad.agent.md`). Init skips existing user files (idempotent); upgrades only overwrite Squad-owned files.

### Git Merge Strategy

`.gitattributes` uses `merge=union` for append-only files (decisions, history, logs, orchestration-log). This enables conflict-free merging of squad state across branches.

### Team File Structure

`team.md` **must** have a section titled exactly `## Members` — GitHub workflows (`squad-heartbeat.yml`, `squad-issue-assign.yml`, `squad-triage.yml`, `sync-squad-labels.yml`) hardcode this header for label automation.

### Branch Naming

Squad branches follow: `squad/{issue-number}-{kebab-case-slug}` (e.g., `squad/42-fix-login-validation`).

### Decision Flow

Agents write decisions to `.squad/decisions/inbox/{agent-name}-{slug}.md`. The Scribe merges them into `.squad/decisions.md`.

## Squad Skills

Read `.squad/skills/squad-conventions/SKILL.md` for the full conventions reference. Key points:

- **Zero runtime dependencies** — Node.js built-ins only (`fs`, `path`, `os`, `child_process`). Never add packages to `dependencies`.
- **Tests** — `node:test` and `node:assert/strict`. Run with `npm test` (`node --test test/`).
- **Error handling** — All user-facing errors go through `fatal(msg)` (red `✗` prefix, exit 1). No raw stack traces.
- **ANSI colors** — Use constants (`GREEN`, `RED`, `DIM`, `BOLD`, `RESET`). Never inline escape codes.
- **Windows compatibility** — Always use `path.join()`. Never hardcode `/` or `\` separators.
