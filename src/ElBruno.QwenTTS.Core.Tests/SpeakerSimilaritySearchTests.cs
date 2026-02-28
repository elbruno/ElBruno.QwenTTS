using ElBruno.QwenTTS.Models;
using Xunit;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests for PERF-1: Top-K heap-based speaker similarity search.
/// Validates correctness, performance, and edge cases.
/// </summary>
public sealed class SpeakerSimilaritySearchTests
{
    [Fact]
    public void FindTopK_IdenticalVector_ReturnsExactMatch()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f, 0.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 1.0f, 0.0f, 0.0f, 0.0f }), // exact match
            ("speaker2", new float[] { 0.0f, 1.0f, 0.0f, 0.0f }),
            ("speaker3", new float[] { 0.0f, 0.0f, 1.0f, 0.0f }),
        };

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 1);

        // Assert
        Assert.Single(matches);
        Assert.Equal("speaker1", matches[0].SpeakerId);
        Assert.True(matches[0].Similarity > 0.999f); // cosine(identical) = 1.0
    }

    [Fact]
    public void FindTopK_ThreeResults_ReturnsInDescendingOrder()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f, 0.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 0.9f, 0.1f, 0.0f, 0.0f }), // high similarity
            ("speaker2", new float[] { 0.5f, 0.5f, 0.5f, 0.0f }), // medium
            ("speaker3", new float[] { 0.0f, 0.0f, 1.0f, 0.0f }), // low
            ("speaker4", new float[] { 1.0f, 0.0f, 0.0f, 0.0f }), // highest (exact)
            ("speaker5", new float[] { 0.7f, 0.3f, 0.0f, 0.0f }), // medium-high
        };

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 3);

        // Assert
        Assert.Equal(3, matches.Length);
        Assert.Equal("speaker4", matches[0].SpeakerId); // highest first
        Assert.True(matches[0].Similarity > matches[1].Similarity);
        Assert.True(matches[1].Similarity > matches[2].Similarity);
    }

    [Fact]
    public void FindTopK_MoreResultsThanAvailable_ReturnsAll()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 1.0f, 0.0f }),
            ("speaker2", new float[] { 0.0f, 1.0f }),
        };

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 10);

        // Assert
        Assert.Equal(2, matches.Length); // only 2 available
    }

    [Fact]
    public void FindTopK_LargeCollection_MaintainsCorrectTopK()
    {
        // Arrange: Create 1000 speakers, top 5 have highest similarities
        var query = new float[] { 1.0f, 0.0f, 0.0f, 0.0f };
        var references = new List<(string, float[])>();
        
        // Top 5 with known high similarities
        references.Add(("top1", new float[] { 1.0f, 0.0f, 0.0f, 0.0f })); // 1.0
        references.Add(("top2", new float[] { 0.99f, 0.01f, 0.0f, 0.0f })); // ~0.99
        references.Add(("top3", new float[] { 0.98f, 0.02f, 0.0f, 0.0f })); // ~0.98
        references.Add(("top4", new float[] { 0.97f, 0.03f, 0.0f, 0.0f })); // ~0.97
        references.Add(("top5", new float[] { 0.96f, 0.04f, 0.0f, 0.0f })); // ~0.96
        
        // 995 random lower-similarity speakers
        var random = new Random(42);
        for (int i = 0; i < 995; i++)
        {
            var vec = new float[] { 
                (float)random.NextDouble() * 0.5f, // 0-0.5 range
                (float)random.NextDouble(),
                (float)random.NextDouble(),
                (float)random.NextDouble()
            };
            references.Add(($"random{i}", vec));
        }

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 5);

        // Assert
        Assert.Equal(5, matches.Length);
        Assert.Equal("top1", matches[0].SpeakerId);
        Assert.Equal("top2", matches[1].SpeakerId);
        Assert.Equal("top3", matches[2].SpeakerId);
        Assert.Equal("top4", matches[3].SpeakerId);
        Assert.Equal("top5", matches[4].SpeakerId);
    }

    [Fact]
    public void FindTopK_NormalizedAndUnnormalized_ProduceSameRanking()
    {
        // Arrange
        var query = new float[] { 2.0f, 0.0f, 0.0f };
        var queryNormalized = new float[] { 1.0f, 0.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 3.0f, 0.0f, 0.0f }), // same direction
            ("speaker2", new float[] { 0.0f, 2.0f, 0.0f }),
            ("speaker3", new float[] { 1.5f, 1.5f, 0.0f }),
        };

        // Act
        var matches1 = SpeakerSimilaritySearch.FindTopK(query, references, k: 3);
        var matches2 = SpeakerSimilaritySearch.FindTopK(queryNormalized, references, k: 3);

        // Assert — same ranking regardless of normalization
        Assert.Equal(matches1[0].SpeakerId, matches2[0].SpeakerId);
        Assert.Equal(matches1[1].SpeakerId, matches2[1].SpeakerId);
        Assert.Equal(matches1[2].SpeakerId, matches2[2].SpeakerId);
    }

    [Fact]
    public void FindTopK_ZeroVector_HandlesGracefully()
    {
        // Arrange
        var query = new float[] { 0.0f, 0.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 1.0f, 0.0f, 0.0f }),
            ("speaker2", new float[] { 0.0f, 1.0f, 0.0f }),
        };

        // Act — should not throw
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 2);

        // Assert — returns results (similarity will be 0 or NaN, but no crash)
        Assert.Equal(2, matches.Length);
    }

    [Fact]
    public void FindTopK_DimensionMismatch_ThrowsArgumentException()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 1.0f, 0.0f, 0.0f }), // dimension mismatch
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => 
            SpeakerSimilaritySearch.FindTopK(query, references, k: 1));
    }

    [Fact]
    public void FindTopK_InvalidK_ThrowsArgumentException()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f };
        var references = new[]
        {
            ("speaker1", new float[] { 1.0f, 0.0f }),
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => 
            SpeakerSimilaritySearch.FindTopK(query, references, k: 0));
        Assert.Throws<ArgumentException>(() => 
            SpeakerSimilaritySearch.FindTopK(query, references, k: -1));
    }

    [Fact]
    public void FindTopK_EmptyReferences_ReturnsEmptyArray()
    {
        // Arrange
        var query = new float[] { 1.0f, 0.0f };
        var references = Array.Empty<(string, float[])>();

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 5);

        // Assert
        Assert.Empty(matches);
    }

    [Fact]
    public void FindTopK_HighDimensionalEmbeddings_WorksCorrectly()
    {
        // Arrange — simulate realistic 1024-dim embeddings
        var random = new Random(42);
        var query = new float[1024];
        for (int i = 0; i < 1024; i++)
            query[i] = (float)random.NextDouble();

        var references = new List<(string, float[])>();
        for (int i = 0; i < 100; i++)
        {
            var embedding = new float[1024];
            for (int j = 0; j < 1024; j++)
                embedding[j] = (float)random.NextDouble();
            references.Add(($"speaker{i}", embedding));
        }

        // Act
        var matches = SpeakerSimilaritySearch.FindTopK(query, references, k: 10);

        // Assert
        Assert.Equal(10, matches.Length);
        // Verify descending order
        for (int i = 0; i < matches.Length - 1; i++)
        {
            Assert.True(matches[i].Similarity >= matches[i + 1].Similarity);
        }
    }

    /// <summary>
    /// Benchmark baseline: Measures Top-K lookup time for documentation.
    /// Not a performance test — just captures timing for future comparison.
    /// </summary>
    [Fact]
    public void Benchmark_TopK_1000Speakers_K10()
    {
        // Arrange
        var random = new Random(42);
        var query = new float[1024];
        for (int i = 0; i < 1024; i++)
            query[i] = (float)random.NextDouble();

        var references = new List<(string, float[])>();
        for (int i = 0; i < 1000; i++)
        {
            var embedding = new float[1024];
            for (int j = 0; j < 1024; j++)
                embedding[j] = (float)random.NextDouble();
            references.Add(($"speaker{i}", embedding));
        }

        // Act — warmup + timed run
        SpeakerSimilaritySearch.FindTopK(query, references, k: 10); // warmup
        
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 100; i++)
        {
            SpeakerSimilaritySearch.FindTopK(query, references, k: 10);
        }
        sw.Stop();

        // Assert — document baseline (no hard threshold, just log)
        var avgMs = sw.ElapsedMilliseconds / 100.0;
        Assert.True(avgMs < 1000); // sanity check: should be well under 1 second per query
        
        // Output for manual review (appears in test logs)
        Console.WriteLine($"PERF-1 Baseline: Top-10 from 1000 speakers (1024-dim) = {avgMs:F2} ms average");
    }
}
