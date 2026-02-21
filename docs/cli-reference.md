# CLI Reference

## Usage

```bash
dotnet run --project src/QwenTTS -- --model-dir <path> --text "<text>" [options]
```

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | *(required)* | Path to downloaded model directory |
| `--text` | *(required)* | Text to synthesize |
| `--speaker` | `Ryan` | Speaker voice name (see below) |
| `--language` | `auto` | Language: `english`, `chinese`, `auto`, etc. |
| `--output` | `output.wav` | Output WAV file path |
| `--instruct` | *(none)* | Voice style instruction (e.g., "speak happily") |

## Available Speakers

| Speaker | Notes |
|---------|-------|
| `ryan` | English male |
| `serena` | English female |
| `vivian` | English female |
| `aiden` | English male |
| `eric` | Sichuan dialect |
| `dylan` | Beijing dialect |
| `uncle_fu` | Chinese male |
| `ono_anna` | Japanese female |
| `sohee` | Korean female |

## Examples

### English

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hello, this is a test." --speaker ryan --language english --output hello.wav
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Welcome to the future of speech synthesis." --speaker serena --output welcome.wav
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "This is Aiden speaking with excitement." --speaker aiden --instruct "speak with excitement" --output excited.wav
```

### Spanish

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hola, esta es una prueba de texto a voz." --speaker ryan --language spanish --output hola.wav
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Bienvenidos al futuro de la sintesis de voz." --speaker serena --language spanish --output bienvenidos.wav
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hablando con emocion y energia." --speaker aiden --language spanish --instruct "speak with excitement" --output emocion.wav
```

### File Reader (batch text/SRT to audio)

See [File Reader](file-reader.md) for full details.

```bash
dotnet run --project src/QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/hello_demo.txt --speaker ryan --language english --output-dir output/hello
dotnet run --project src/QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/hola_demo.txt --speaker ryan --language spanish --output-dir output/hola
dotnet run --project src/QwenTTS.FileReader -- --model-dir python/onnx_runtime --input samples/demo_subtitles.srt --speaker serena --output-dir output/subtitles
```
