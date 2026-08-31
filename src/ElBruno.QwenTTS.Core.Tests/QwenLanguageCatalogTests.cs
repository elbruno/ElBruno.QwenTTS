using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

public class QwenLanguageCatalogTests
{
    [Fact]
    public void SupportedLanguages_IncludeRussian()
    {
        Assert.Contains(QwenLanguageCatalog.Options, option => option.Value == "russian");
        Assert.True(QwenLanguageCatalog.IsSupported("russian"));
    }

    [Fact]
    public void SupportedLanguages_IncludeAutoAndCommonLanguages()
    {
        Assert.Contains(QwenLanguageCatalog.Options, option => option.Value == "auto");
        Assert.Contains(QwenLanguageCatalog.Options, option => option.Value == "english");
        Assert.Contains(QwenLanguageCatalog.Options, option => option.Value == "spanish");
    }

    [Theory]
    [InlineData("german")]
    [InlineData("french")]
    [InlineData("portuguese")]
    [InlineData("italian")]
    public void SupportedLanguages_IncludeAdditionalModelLanguages(string language)
    {
        Assert.Contains(QwenLanguageCatalog.Options, option => option.Value == language);
        Assert.True(QwenLanguageCatalog.IsSupported(language));
    }

    [Fact]
    public void Options_MatchTheTenLanguagesSupportedByTheModel()
    {
        string[] expected =
        [
            "auto", "english", "spanish", "chinese", "japanese",
            "korean", "russian", "german", "french", "portuguese", "italian"
        ];

        Assert.Equal(expected.Order(), QwenLanguageCatalog.Options.Select(o => o.Value).Order());
        Assert.Equal(expected.Length - 1, QwenLanguageCatalog.SupportedLanguages.Count);
    }

    [Fact]
    public void Options_HaveUniqueValuesAndNonEmptyLabels()
    {
        Assert.Equal(
            QwenLanguageCatalog.Options.Count,
            QwenLanguageCatalog.Options.Select(o => o.Value).Distinct(StringComparer.OrdinalIgnoreCase).Count());

        Assert.All(QwenLanguageCatalog.Options, option => Assert.False(string.IsNullOrWhiteSpace(option.Label)));
    }

    [Fact]
    public void IsSupported_IsCaseInsensitiveAndExcludesAuto()
    {
        Assert.True(QwenLanguageCatalog.IsSupported("Italian"));
        Assert.True(QwenLanguageCatalog.IsSupported("GERMAN"));
        Assert.False(QwenLanguageCatalog.IsSupported("auto"));
        Assert.False(QwenLanguageCatalog.IsSupported("klingon"));
        Assert.False(QwenLanguageCatalog.IsSupported(null));
        Assert.False(QwenLanguageCatalog.IsSupported("  "));
    }
}
