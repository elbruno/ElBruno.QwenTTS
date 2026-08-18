using System.Diagnostics;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>OpenTelemetry activity helpers for local Qwen TTS operations.</summary>
public static class QwenTtsTelemetry
{
    /// <summary>The activity source used by Qwen TTS and voice cloning workflows.</summary>
    public static readonly ActivitySource ActivitySource = new("ElBruno.QwenTTS");

    /// <summary>Starts a model download activity without recording a local path or file name.</summary>
    public static Activity? StartModelDownload(string modelKind) =>
        Start("gen_ai.model.download", "model.download", modelKind);

    /// <summary>Starts a text-to-speech workflow activity without recording input text or output paths.</summary>
    public static Activity? StartTextToSpeech() =>
        Start("gen_ai.text_to_speech", "text_to_speech", "custom-voice");

    /// <summary>Starts a voice-cloning workflow activity without recording reference content.</summary>
    public static Activity? StartVoiceCloning(string operation) =>
        Start($"gen_ai.voice_clone.{operation}", $"voice_clone.{operation}", "base");

    private static Activity? Start(string activityName, string operationName, string modelKind)
    {
        var activity = ActivitySource.StartActivity(activityName, ActivityKind.Internal);
        activity?.SetTag("gen_ai.operation.name", operationName);
        activity?.SetTag("qwen.model.kind", modelKind);
        return activity;
    }
}
