# Podcast Script Generator Prompt

Use this prompt with ChatGPT, Claude, or any LLM to generate a podcast script that can be directly used with the Qwen TTS Podcast Generator.

The parser supports two formats:
- **`--- NN` delimiter format** (recommended) — simple, LLM-friendly, unambiguous
- **`## Segment NN` markdown format** — renders nicely in markdown viewers

Both formats support implicit text: everything after the metadata fields is the spoken text.

---

## Prompt — Generic (from a topic)

```
Generate a podcast conversation script between two hosts about [YOUR TOPIC HERE].

Rules:
- Use exactly this format for each segment (the --- separator and field names are required)
- Available speakers: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee
- Available languages: english, spanish, chinese, japanese, korean
- Keep each text segment under 200 words for best audio quality
- Alternate between speakers naturally
- Include an intro and outro
- At the beginning each speaker must introduce themselves

Format (copy exactly — text goes after the metadata fields):

--- 01
speaker: [speaker_name]
language: [language]

[What this person says goes here as plain text]

--- 02
speaker: [speaker_name]
language: [language]

[What this person says goes here as plain text]

Generate 8-12 segments for a ~5 minute podcast episode.
Topic: [YOUR TOPIC HERE]
```

---

## Prompt — Context-Based (from existing content)

```
Based on this content, generate a podcast conversation script between two or three hosts about this topic.

Rules:
- Use exactly this format for each segment (the --- separator and field names are required)
- Available speakers: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee
- Available languages: english, spanish, chinese, japanese, korean
- Keep each text segment under 200 words for best audio quality
- Alternate between speakers naturally
- Include an intro and outro

Format (copy exactly — text goes after the metadata fields):

--- 01
speaker: [speaker_name]
language: [language]

[What this person says goes here as plain text]

--- 02
speaker: [speaker_name]
language: [language]

[What this person says goes here as plain text]

Generate the necessary segments for a ~10 minute podcast episode.
At the beginning each speaker must introduce themselves.

Content:
[PASTE YOUR CONTENT HERE]
```

---

## Example: English podcast about AI

```
Generate a podcast conversation script between two hosts about the future of AI agents in software development.

Rules:
- Use exactly this format for each segment (the --- separator and field names are required)
- Use speakers "ryan" and "serena" alternating
- Language: english
- Keep each text segment under 200 words for best audio quality
- Include an intro and outro
- At the beginning each speaker must introduce themselves

Format (copy exactly):

--- 01
speaker: ryan
language: english

[intro text here]

--- 02
speaker: serena
language: english

[response text here]

Generate 10 segments for a ~5 minute podcast episode.
Topic: The future of AI agents in software development
```

## Example: Spanish podcast

```
Genera un guion de podcast conversacional entre dos presentadores sobre inteligencia artificial y agentes de IA.

Reglas:
- Usa exactamente este formato para cada segmento
- Usa los presentadores "ryan" y "serena" alternando
- Idioma: spanish
- Mantén cada segmento de texto bajo 200 palabras
- Incluye una intro y un cierre
- Al inicio cada presentador debe presentarse

Formato (copiar exactamente):

--- 01
speaker: ryan
language: spanish

[texto de intro aquí]

--- 02
speaker: serena
language: spanish

[texto de respuesta aquí]

Genera 10 segmentos para un episodio de podcast de ~5 minutos.
Tema: El futuro de los agentes de IA en el desarrollo de software
```
