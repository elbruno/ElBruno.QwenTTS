# Podcast Generator

The **QwenTTS.Podcast** Blazor Server application lets you generate full podcast episodes from structured multi-speaker script files.

## Quick Start

1. Run the podcast app: `dotnet run --project src/QwenTTS.Podcast`
2. Navigate to [http://localhost:5217](http://localhost:5217)
3. Upload a podcast script file (see format below)
4. Click **Generate Podcast Episode**
5. Download the merged audio file

## Script Format

The parser accepts two formats. Both support implicit text (everything after metadata fields is the spoken content).

### Format 1: `--- NN` delimiter (recommended)

```
--- 01
speaker: ryan
language: english

Welcome to the show! Today we're talking about AI agents.

--- 02
speaker: serena
language: english

That's right Ryan. AI agents are transforming software development.
```

### Format 2: `## Segment NN` markdown

```
## Segment 01
**Speaker:** ryan
**Language:** english

Welcome to the show! Today we're talking about AI agents.

## Segment 02
**Speaker:** serena
**Language:** english

That's right Ryan. AI agents are transforming software development.
```

> **Backward compatibility:** The explicit `text:` prefix from the original format is still supported.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `speaker` | Yes | Speaker voice name (ryan, serena, vivian, aiden, etc.) |
| `language` | No | Language (english, spanish, chinese, etc.). Defaults to english |
| `instructions` | No | Voice style instruction (e.g., "speak calmly") |
| text (implicit) | Yes | Everything after the metadata fields. Keep under 200 words per segment |

## Generating Scripts with AI

Use the prompt template in `samples/podcast_prompt_template.md` to generate podcast scripts with ChatGPT, Claude, or any LLM. The template includes:

- Instructions for the LLM to produce the correct format
- English and Spanish prompt examples
- Available speaker and language lists

## Features

- **Segment preview table** — shows parsed segments with speaker, language, text preview, and per-segment status
- **Progress bar** — tracks generation progress across all segments
- **Console log** — real-time pipeline events (docked at bottom)
- **Silence gaps** — 0.5s silence automatically inserted between segments for natural pacing
- **Download full episode** — single merged WAV file
- **Individual segments** — expandable section to download individual segment WAV files

## Sample Files

- `samples/podcast_ai_agents.txt` — 12-segment English podcast about AI agents
- `samples/podcast_prompt_template.md` — LLM prompt template for generating scripts
