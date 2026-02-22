namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Pre-defined voice presets for Qwen3-TTS CustomVoice model.
/// Use <see cref="ToSpeakerName"/> to convert to the string speaker name.
/// </summary>
public enum QwenVoicePreset
{
    Ryan,
    Serena,
    Vivian,
    Aiden,
    Eric,
    Dylan,
    UncleFu,
    OnoAnna,
    Sohee
}

/// <summary>
/// Extension methods for <see cref="QwenVoicePreset"/>.
/// </summary>
public static class QwenVoicePresetExtensions
{
    /// <summary>
    /// Converts a voice preset to the speaker name string used by the model.
    /// </summary>
    public static string ToSpeakerName(this QwenVoicePreset preset) => preset switch
    {
        QwenVoicePreset.Ryan => "ryan",
        QwenVoicePreset.Serena => "serena",
        QwenVoicePreset.Vivian => "vivian",
        QwenVoicePreset.Aiden => "aiden",
        QwenVoicePreset.Eric => "eric",
        QwenVoicePreset.Dylan => "dylan",
        QwenVoicePreset.UncleFu => "uncle_fu",
        QwenVoicePreset.OnoAnna => "ono_anna",
        QwenVoicePreset.Sohee => "sohee",
        _ => preset.ToString().ToLowerInvariant()
    };
}
