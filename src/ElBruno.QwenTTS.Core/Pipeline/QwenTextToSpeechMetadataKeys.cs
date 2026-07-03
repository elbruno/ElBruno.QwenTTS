namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Additional property names used by the Microsoft.Extensions.AI text-to-speech adapter.
/// </summary>
public static class QwenTextToSpeechMetadataKeys
{
    public const string Variant = "elbruno.qwentts.variant";
    public const string Speaker = "elbruno.qwentts.speaker";
    public const string Language = "elbruno.qwentts.language";
    public const string Instruct = "elbruno.qwentts.instruct";
    public const string VoiceCloning = "elbruno.qwentts.voice_cloning";
    public const string ExecutionProvider = "elbruno.qwentts.execution_provider";
}
