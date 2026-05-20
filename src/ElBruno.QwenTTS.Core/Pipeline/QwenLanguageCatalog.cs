namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Shared catalog of supported TTS language choices.
/// </summary>
public sealed record QwenLanguageOption(string Value, string Label);

public static class QwenLanguageCatalog
{
    public static readonly IReadOnlyList<QwenLanguageOption> Options =
    [
        new("auto", "Auto"),
        new("english", "English"),
        new("spanish", "Spanish"),
        new("chinese", "Chinese"),
        new("japanese", "Japanese"),
        new("korean", "Korean"),
        new("russian", "Russian")
    ];

    public static readonly IReadOnlySet<string> SupportedLanguages =
        Options.Where(option => option.Value != "auto")
            .Select(option => option.Value)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

    public static bool IsSupported(string? language) =>
        !string.IsNullOrWhiteSpace(language) && SupportedLanguages.Contains(language);
}
