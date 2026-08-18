namespace ElBruno.QwenTTS.BlazorComponents.Models;

/// <summary>
/// Limits browser-provided reference audio to fit within the SignalR message limit after base64 encoding.
/// </summary>
public static class VoiceCloneReferenceAudioLimits
{
    public const long MaxPayloadBytes = 1 * 1024 * 1024;
}
