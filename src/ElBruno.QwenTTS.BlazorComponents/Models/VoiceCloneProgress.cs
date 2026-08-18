namespace ElBruno.QwenTTS.BlazorComponents.Models;

/// <summary>
/// Describes host-owned voice-cloning workflow progress.
/// </summary>
public sealed record VoiceCloneProgress(
    string Message,
    double? Percentage = null,
    bool IsIndeterminate = false);
