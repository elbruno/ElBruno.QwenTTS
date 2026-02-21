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

```bash
dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Hello, this is a test." --speaker ryan --language english --output hello.wav

dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "Welcome to the future of speech synthesis." --speaker serena --output welcome.wav

dotnet run --project src/QwenTTS -- --model-dir python/onnx_runtime --text "This is Aiden speaking with excitement." --speaker aiden --instruct "speak with excitement" --output excited.wav
```
