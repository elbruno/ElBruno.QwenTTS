using ElBruno.QwenTTS.Models;

namespace ElBruno.QwenTTS.Core.Tests;

public class TextTokenizerTests
{
    [Fact]
    public void Encode_UsesByteLevelBpeAndPreservesSpecialTokens()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"qwen_tokenizer_{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "vocab.json"),
                """{"h":0,"i":1,"hi":2,"\u0120":3,"\u0120hi":4,"\u00c3":5,"\u00a9":6}""");
            File.WriteAllText(Path.Combine(directory, "merges.txt"), "#version: 0.2\nh i\n\u0120 hi\n");
            using var tokenizer = new TextTokenizer(directory);

            Assert.Equal(new[] { 2, 4, 3, 5, 6, TextTokenizer.ImEndId },
                tokenizer.Encode("hi hi é<|im_end|>"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
