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
}
