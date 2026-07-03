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
    public void Constructor_AcceptsMaxConcurrency()
    {
        using var client = new QwenTextToSpeechClient(maxConcurrency: 2);
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
        Assert.NotNull(response.Metrics);
        Assert.Equal(TimeSpan.Zero, response.Metrics.QueueLatency);
        Assert.Equal(TimeSpan.Zero, response.Metrics.FirstAudioLatency);
        Assert.Equal(TimeSpan.Zero, response.Metrics.TotalLatency);
    }

    [Fact]
    public void TextToSpeechStreamingUpdate_CanBeCreated()
    {
        var open = new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.SessionOpen,
            MediaType = "audio/wav",
            SampleRate = 24000,
            Channels = 1,
            BitsPerSample = 16,
            IsProgressive = false
        };
        var chunk = new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.AudioChunk,
            AudioData = [1, 2, 3],
            SampleRate = 24000,
            MediaType = "audio/wav",
            Channels = 1,
            BitsPerSample = 16,
            IsProgressive = false
        };
        var close = new TextToSpeechStreamingUpdate
        {
            Kind = TextToSpeechUpdateKind.SessionClose,
            Metrics = new TtsSynthesisMetrics { OutputSamples = 3 }
        };

        Assert.Equal(TextToSpeechUpdateKind.SessionOpen, open.Kind);
        Assert.Null(open.AudioData);
        Assert.Equal("audio/wav", open.MediaType);
        Assert.Equal(1, open.Channels);
        Assert.Equal(16, open.BitsPerSample);
        Assert.False(open.IsProgressive);
        Assert.Equal(TextToSpeechUpdateKind.AudioChunk, chunk.Kind);
        Assert.NotNull(chunk.AudioData);
        Assert.Equal(24000, chunk.SampleRate);
        Assert.Equal("audio/wav", chunk.MediaType);
        Assert.Equal(TextToSpeechUpdateKind.SessionClose, close.Kind);
        Assert.Equal(3, close.Metrics?.OutputSamples);
    }

    [Fact]
    public async Task SynthesizeStreamingAsync_UsesPipelineStreamingUpdates()
    {
        using var client = new QwenTextToSpeechClient(
            defaultVoice: "ryan",
            defaultLanguage: "english",
            modelDir: null,
            repoId: null,
            variant: QwenModelVariant.Qwen06B,
            sessionOptionsFactory: null,
            vocoderSessionOptionsFactory: null,
            maxConcurrency: 1,
            pipelineFactory: _ => Task.FromResult<ITtsPipeline>(new FakeTtsPipeline()));

        var updates = new List<TextToSpeechStreamingUpdate>();
        await foreach (var update in client.SynthesizeStreamingAsync("Hello from streaming"))
            updates.Add(update);

        Assert.Collection(
            updates,
            update =>
            {
                Assert.Equal(TextToSpeechUpdateKind.SessionOpen, update.Kind);
                Assert.Equal("audio/wav", update.MediaType);
                Assert.Equal(24000, update.SampleRate);
                Assert.False(update.IsProgressive);
            },
            update => Assert.Equal(new byte[] { 82, 73, 70, 70 }, update.AudioData![..4]),
            update => Assert.Equal(new byte[] { 1, 2, 3, 4 }, update.AudioData),
            update =>
            {
                Assert.Equal(TextToSpeechUpdateKind.SessionClose, update.Kind);
                Assert.Equal(4, update.Metrics?.OutputSamples);
            });
    }

    [Fact]
    public async Task SynthesizeStreamingAsync_StopsAfterCancellation()
    {
        using var client = new QwenTextToSpeechClient(
            defaultVoice: "ryan",
            defaultLanguage: "english",
            modelDir: null,
            repoId: null,
            variant: QwenModelVariant.Qwen06B,
            sessionOptionsFactory: null,
            vocoderSessionOptionsFactory: null,
            maxConcurrency: 1,
            pipelineFactory: _ => Task.FromResult<ITtsPipeline>(new FakeTtsPipeline()));

        using var cancellationTokenSource = new CancellationTokenSource();
        var updates = new List<TextToSpeechStreamingUpdate>();

        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (var update in client.SynthesizeStreamingAsync(
                "Cancel after open",
                cancellationToken: cancellationTokenSource.Token))
            {
                updates.Add(update);
                cancellationTokenSource.Cancel();
            }
        });

        Assert.Single(updates);
        Assert.Equal(TextToSpeechUpdateKind.SessionOpen, updates[0].Kind);
    }

    [Fact]
    public async Task SynthesizeStreamingAsync_ThrowsWhenDisposed()
    {
        var client = new QwenTextToSpeechClient();
        client.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(async () =>
        {
            await foreach (var _ in client.SynthesizeStreamingAsync("Hello")) { }
        });
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
            opts.MaxConcurrency = 2;
        });

        var descriptor = Assert.Single(services, d => d.ServiceType == typeof(ITextToSpeechClient));
        Assert.Equal(ServiceLifetime.Singleton, descriptor.Lifetime);
    }

    [Fact]
    public void QwenTtsOptions_MaxConcurrencyDefaultsToOne()
    {
        var options = new QwenTtsOptions();
        Assert.Equal(1, options.MaxConcurrency);
    }

    public void Dispose()
    {
        _client.Dispose();
    }

    private sealed class FakeTtsPipeline : ITtsPipeline
    {
        public IReadOnlyCollection<string> Speakers => ["ryan"];

        public QwenModelVariant ModelVariant => QwenModelVariant.Qwen06B;

        public Task SynthesizeAsync(string text, string speaker, string outputPath, string language = "auto", string? instruct = null, IProgress<string>? progress = null)
            => throw new NotSupportedException();

        public async IAsyncEnumerable<TextToSpeechStreamingUpdate> SynthesizeStreamingAsync(
            string text,
            string speaker,
            string language = "auto",
            string? instruct = null,
            IProgress<string>? progress = null,
            int maxChunkBytes = 32 * 1024,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            yield return new TextToSpeechStreamingUpdate
            {
                Kind = TextToSpeechUpdateKind.SessionOpen,
                MediaType = "audio/wav",
                SampleRate = 24000,
                Channels = 1,
                BitsPerSample = 16,
                IsProgressive = false
            };

            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();

            yield return new TextToSpeechStreamingUpdate
            {
                Kind = TextToSpeechUpdateKind.AudioChunk,
                AudioData =
                [
                    (byte)'R', (byte)'I', (byte)'F', (byte)'F',
                    0, 0, 0, 0
                ],
                MediaType = "audio/wav",
                SampleRate = 24000,
                Channels = 1,
                BitsPerSample = 16,
                IsProgressive = false
            };

            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();

            yield return new TextToSpeechStreamingUpdate
            {
                Kind = TextToSpeechUpdateKind.AudioChunk,
                AudioData = [1, 2, 3, 4],
                MediaType = "audio/wav",
                SampleRate = 24000,
                Channels = 1,
                BitsPerSample = 16,
                IsProgressive = false
            };

            cancellationToken.ThrowIfCancellationRequested();

            yield return new TextToSpeechStreamingUpdate
            {
                Kind = TextToSpeechUpdateKind.SessionClose,
                MediaType = "audio/wav",
                SampleRate = 24000,
                Channels = 1,
                BitsPerSample = 16,
                IsProgressive = false,
                Metrics = new TtsSynthesisMetrics { OutputSamples = 4 }
            };
        }

        public Task<TtsSynthesisMetrics> SynthesizeWithMetricsAsync(
            string text,
            string speaker,
            string outputPath,
            string language = "auto",
            string? instruct = null,
            IProgress<string>? progress = null,
            CancellationToken cancellationToken = default)
            => Task.FromResult(new TtsSynthesisMetrics());

        public void Dispose()
        {
        }
    }
}
