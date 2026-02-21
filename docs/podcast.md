# Podcast Generator

The **Podcast Generator** page lets you generate a full podcast episode from a structured multi-speaker script file.

## Quick Start

1. Run the web app: `dotnet run --project src/QwenTTS.Web`
2. Navigate to [http://localhost:5123/podcast](http://localhost:5123/podcast)
3. Upload a podcast script file (see format below)
4. Click **Generate Podcast Episode**
5. Download the merged audio file

## Script Format

Podcast scripts use `--- NN` delimiters to separate segments. Each segment specifies a speaker, language, voice instructions, and the text to speak:

```
--- 01
speaker: ryan
language: english
instructions: speak with energy and enthusiasm
text: Welcome to the show! Today we're talking about AI agents.

--- 02
speaker: serena
language: english
instructions: speak naturally and conversationally
text: That's right Ryan. AI agents are transforming software development.
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `speaker` | Yes | Speaker voice name (ryan, serena, vivian, aiden, etc.) |
| `language` | No | Language (english, spanish, chinese, etc.). Defaults to english |
| `instructions` | No | Voice style instruction (e.g., "speak calmly") |
| `text` | Yes | The text for this segment. Keep under 200 words per segment |

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
