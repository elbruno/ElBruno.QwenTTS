namespace ElBruno.QwenTTS.BlazorComponents.Models;

/// <summary>
/// Describes an audio sample that can be selected as a voice cloning reference.
/// </summary>
public sealed record VoiceCloningSampleOption(
    string Id,
    string DisplayName,
    string AudioUrl,
    string? Transcript = null,
    string? Description = null);
