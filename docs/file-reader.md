# File Reader

The **ElBruno.QwenTTS.FileReader** console application reads a text or SRT file and generates speech audio for each segment.

## Usage

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input <file> --speaker <name> [options]
```

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | *(required)* | Path to ONNX model directory |
| `--input` | *(required)* | Path to `.txt`, `.md`, or `.srt` file |
| `--speaker` | `Ryan` | Speaker voice name |
| `--language` | `auto` | Language: `english`, `chinese`, `spanish`, `auto`, etc. |
| `--output-dir` | `output` | Directory for generated WAV files |
| `--instruct` | *(none)* | Voice style instruction |

## Supported File Formats

### Plain Text / Markdown (`.txt`, `.md`)

Text files are split into segments by blank lines (paragraphs). Markdown formatting (headings, bold, italic, links, lists) is automatically stripped for cleaner speech output.

### SRT Subtitles (`.srt`)

SRT files are parsed by subtitle entry. Each entry becomes a separate audio file, making it easy to create voiceovers for video content. HTML tags in subtitles are stripped automatically.

## Output

Audio files are saved in the output directory with the naming pattern `{basename}_{NNN}.wav`, where `NNN` is the segment number (zero-padded to 3 digits).

## Examples

### English text file

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/hello_demo.txt --speaker ryan --language english --output-dir output/hello
```

### Spanish text file

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/hola_demo.txt --speaker ryan --language spanish --output-dir output/hola
```

### SRT subtitle file

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/demo_subtitles.srt --speaker serena --language english --output-dir output/subtitles
```

### Podcast script in Spanish

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/NTN_episodio_agentes_IA.md --speaker ryan --language spanish --output-dir output/podcast
```

### With voice style instruction

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/hola_demo.txt --speaker ryan --language spanish --instruct "speak slowly and clearly" --output-dir output/slow
```
