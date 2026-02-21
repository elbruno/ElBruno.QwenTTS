# Podcast Script Generator Prompt

Use this prompt with ChatGPT, Claude, or any LLM to generate a podcast script that can be directly used with the Qwen TTS Podcast Generator.

---

## Prompt

```
Generate a podcast conversation script between two hosts about [YOUR TOPIC HERE].

Rules:
- Use exactly this format for each segment (the --- separator and field names are required):
- Available speakers: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee
- Available languages: english, spanish, chinese, japanese, korean
- Keep each text segment under 200 words for best audio quality
- Alternate between speakers naturally
- Include an intro and outro

Format (copy exactly):

--- 01
speaker: [speaker_name]
language: [language]
instructions: [voice style, e.g. "speak with energy and enthusiasm"]
text: [what this person says]

--- 02
speaker: [speaker_name]
language: [language]
instructions: [voice style]
text: [what this person says]

Generate 8-12 segments for a ~5 minute podcast episode.
Topic: [YOUR TOPIC HERE]
```

---

## Example: English podcast about AI

```
Generate a podcast conversation script between two hosts about the future of AI agents in software development.

Rules:
- Use exactly this format for each segment (the --- separator and field names are required):
- Use speakers "ryan" and "serena" alternating
- Language: english
- Keep each text segment under 200 words for best audio quality
- Include an intro and outro

Format (copy exactly):

--- 01
speaker: ryan
language: english
instructions: speak with energy and enthusiasm
text: [intro text]

--- 02
speaker: serena
language: english
instructions: speak naturally and conversationally
text: [response text]

Generate 10 segments for a ~5 minute podcast episode.
Topic: The future of AI agents in software development
```

## Example: Spanish podcast

```
Genera un guion de podcast conversacional entre dos presentadores sobre inteligencia artificial y agentes de IA.

Reglas:
- Usa exactamente este formato para cada segmento:
- Usa los presentadores "ryan" y "serena" alternando
- Idioma: spanish
- Mantén cada segmento de texto bajo 200 palabras
- Incluye una intro y un cierre

Formato (copiar exactamente):

--- 01
speaker: ryan
language: spanish
instructions: habla con energia y entusiasmo
text: [texto de intro]

--- 02
speaker: serena
language: spanish
instructions: habla de forma natural y conversacional
text: [texto de respuesta]

Genera 10 segmentos para un episodio de podcast de ~5 minutos.
Tema: El futuro de los agentes de IA en el desarrollo de software
```
