namespace ElBruno.QwenTTS.BlazorComponents.Models;

/// <summary>
/// Audio supplied by a browser for use as a voice-cloning reference.
/// </summary>
public sealed record VoiceCloneReferenceAudio(
    string FileName,
    string ContentType,
    byte[] Content,
    VoiceCloneReferenceAudioSource Source);

/// <summary>
/// Identifies how a voice-cloning reference was acquired.
/// </summary>
public enum VoiceCloneReferenceAudioSource
{
    Upload,
    Recording
}
