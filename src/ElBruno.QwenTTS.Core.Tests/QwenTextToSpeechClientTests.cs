using ElBruno.QwenTTS.Pipeline;
using Microsoft.Extensions.DependencyInjection;

namespace ElBruno.QwenTTS.Core.Tests;

public class QwenTextToSpeechClientTests : IDisposable
{
    private readonly QwenTextToSpeechClient _client;

    public QwenTextToSpeechClientTests()
    {
        _client = new QwenTextToSpeechClient(
            defaultVoice: "ryan",
            defaultLanguage: "english");
    }

    [Fact]
    public void Constructor_SetsDefaults()
    {
        using var client = new QwenTextToSpeechClient();
        // Should not throw — defaults are valid
        Assert.NotNull(client);
    }

    [Fact]
    public void Constructor_AcceptsCustomValues()
    {
        using var client = new QwenTextToSpeechClient(
            defaultVoice: "serena",
            defaultLanguage: "spanish",
            modelDir: "/tmp/test-models");
        Assert.NotNull(client);
    }

    [Fact]
    public void Constructor_AcceptsRussianLanguage()
    {
        using var client = new QwenTextToSpeechClient(defaultLanguage: "russian");
        Assert.NotNull(client);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var client = new QwenTextToSpeechClient();
        client.Dispose();
        client.Dispose(); // Should not throw
    }

    [Fact]
    public async Task SynthesizeToMemoryAsync_ThrowsWhenDisposed()
    {
        var client = new QwenTextToSpeechClient();
        client.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => client.SynthesizeToMemoryAsync("Hello"));
    }

    [Fact]
    public async Task SynthesizeToMemoryAsync_ThrowsOnNullText()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(
            () => _client.SynthesizeToMemoryAsync(null!));
    }

    [Fact]
    public async Task SynthesizeToMemoryAsync_ThrowsOnEmptyText()
    {
        await Assert.ThrowsAsync<ArgumentException>(
            () => _client.SynthesizeToMemoryAsync(""));
    }

    [Fact]
    public async Task SynthesizeToMemoryAsync_ThrowsOnWhitespaceText()
    {
        await Assert.ThrowsAsync<ArgumentException>(
            () => _client.SynthesizeToMemoryAsync("   "));
    }

    [Fact]
    public void TextToSpeechOptions_DefaultsAreNull()
    {
        var options = new TextToSpeechOptions();
        Assert.Null(options.VoiceId);
        Assert.Null(options.Language);
        Assert.Null(options.Instruct);
        Assert.Null(options.ModelId);
    }

    [Fact]
    public void TextToSpeechResponse_HasCorrectDefaults()
    {
        var response = new TextToSpeechResponse { AudioData = [1, 2, 3] };
        Assert.Equal("audio/wav", response.MediaType);
        Assert.Equal(24000, response.SampleRate);
        Assert.Equal("qwen3-tts", response.ModelId);
    }

    [Fact]
    public void TextToSpeechStreamingUpdate_CanBeCreated()
    {
        var open = new TextToSpeechStreamingUpdate { Kind = TextToSpeechUpdateKind.SessionOpen };
        var chunk = new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.AudioChunk,
            AudioData = [1, 2, 3],
            SampleRate = 24000
        };
        var close = new TextToSpeechStreamingUpdate { Kind = TextToSpeechUpdateKind.SessionClose };

        Assert.Equal(TextToSpeechUpdateKind.SessionOpen, open.Kind);
        Assert.Null(open.AudioData);
        Assert.Equal(TextToSpeechUpdateKind.AudioChunk, chunk.Kind);
        Assert.NotNull(chunk.AudioData);
        Assert.Equal(24000, chunk.SampleRate);
        Assert.Equal(TextToSpeechUpdateKind.SessionClose, close.Kind);
    }

    [Fact]
    public void AddQwenTextToSpeechClient_RegistersSingleton()
    {
        var services = new ServiceCollection();
        services.AddQwenTextToSpeechClient();

        var descriptor = Assert.Single(services, d => d.ServiceType == typeof(ITextToSpeechClient));
        Assert.Equal(ServiceLifetime.Singleton, descriptor.Lifetime);
    }

    [Fact]
    public void AddQwenTextToSpeechClient_AcceptsConfiguration()
    {
        var services = new ServiceCollection();
        services.AddQwenTextToSpeechClient(opts =>
        {
            opts.ModelPath = "/tmp/models";
            opts.ExecutionProvider = ExecutionProvider.Cpu;
        });

        var descriptor = Assert.Single(services, d => d.ServiceType == typeof(ITextToSpeechClient));
        Assert.Equal(ServiceLifetime.Singleton, descriptor.Lifetime);
    }

    public void Dispose()
    {
        _client.Dispose();
    }
}
