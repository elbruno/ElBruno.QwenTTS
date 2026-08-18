namespace ElBruno.QwenTTS.BlazorComponents.Models;

/// <summary>
/// A host-owned voice-cloning inference request.
/// </summary>
public sealed record VoiceCloneRequest(
    string Text,
    VoiceCloneReferenceAudio ReferenceAudio,
    string? ReferenceTranscript);
