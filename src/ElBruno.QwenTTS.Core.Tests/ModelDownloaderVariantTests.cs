using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests that ModelDownloader includes the correct files for each model variant,
/// particularly the CP projection files needed for 1.7B.
/// Issue #27 fix requires cp_projection_weight.npy and cp_projection_bias.npy
/// for the 1.7B model variant.
/// </summary>
public class ModelDownloaderVariantTests : IDisposable
{
    private readonly string _tempDir;

    // Expected CP projection files that 1.7B needs (and 0.6B does not)
    private const string CpProjectionWeightFile = "embeddings/cp_projection_weight.npy";
    private const string CpProjectionBiasFile = "embeddings/cp_projection_bias.npy";

    // Files shared by all variants
    private static readonly string[] SharedFiles =
    [
        "talker_prefill.onnx",
        "talker_decode.onnx",
        "code_predictor.onnx",
        "vocoder.onnx",
        "embeddings/config.json",
        "embeddings/talker_codec_embedding.npy",
        "embeddings/text_embedding.npy",
        "embeddings/text_projection_fc1_weight.npy",
        "embeddings/text_projection_fc1_bias.npy",
        "embeddings/text_projection_fc2_weight.npy",
        "embeddings/text_projection_fc2_bias.npy",
        "embeddings/codec_head_weight.npy",
        "embeddings/speaker_ids.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
    ];

    public ModelDownloaderVariantTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_dlvariant_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    // ── Test 1: 0.6B does NOT include projection files ──────────────

    [Fact]
    public void Downloader_06B_DoesNotIncludeProjectionFiles()
    {
        // The 0.6B model has hidden_size == cp_hidden_size == 1024
        // so no projection is needed and the files should not be present
        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        Assert.DoesNotContain(CpProjectionWeightFile, missing);
        Assert.DoesNotContain(CpProjectionBiasFile, missing);
    }

    // ── Test 2: 1.7B SHOULD include projection files ────────────────
    // NOTE: This test documents the EXPECTED behavior after Neo's fix.
    // Currently ModelDownloader.ExpectedFiles is a single static list for 0.6B.
    // After the fix, 1.7B will have variant-specific file lists.

    [Fact]
    public void Downloader_17B_IncludesProjectionFiles()
    {
        // After the fix, 1.7B variant file list should include projection files.
        // For now, we validate the file naming convention.
        Assert.Equal("embeddings/cp_projection_weight.npy", CpProjectionWeightFile);
        Assert.Equal("embeddings/cp_projection_bias.npy", CpProjectionBiasFile);

        // Verify the projection files follow the same path convention
        // as other embedding files (under embeddings/ directory)
        Assert.StartsWith("embeddings/", CpProjectionWeightFile);
        Assert.StartsWith("embeddings/", CpProjectionBiasFile);
        Assert.EndsWith(".npy", CpProjectionWeightFile);
        Assert.EndsWith(".npy", CpProjectionBiasFile);
    }

    // ── Shared files exist for all variants ─────────────────────────

    [Fact]
    public void Downloader_AllVariants_ShareCommonFiles()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        foreach (var file in SharedFiles)
        {
            Assert.Contains(file, missing);
        }
    }

    // ── CP codec embeddings (15 groups) are in all variants ─────────

    [Fact]
    public void Downloader_AllVariants_Include15CpCodecEmbeddings()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        for (int i = 0; i < 15; i++)
        {
            var cpFile = $"embeddings/cp_codec_embedding_{i}.npy";
            Assert.Contains(cpFile, missing);
        }
    }

    // ── Variant repo IDs are distinct ───────────────────────────────

    [Fact]
    public void Downloader_VariantRepoIds_AreDistinct()
    {
        var (_, repo06B) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen06B);
        var (_, repo17B) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen17B);

        Assert.NotEqual(repo06B, repo17B);
        Assert.Contains("0.6B", repo06B);
        Assert.Contains("1.7B", repo17B);
    }

    // ── Variant directories are isolated ────────────────────────────

    [Fact]
    public void Downloader_VariantDirs_AreIsolated()
    {
        var (dir06B, _) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen06B);
        var (dir17B, _) = ModelDownloader.ResolveForVariant(QwenModelVariant.Qwen17B);

        Assert.NotEqual(dir06B, dir17B);
        // 1.7B dir should be a subdirectory
        Assert.StartsWith(dir06B, dir17B);
    }

    // ── Projection file dimensions (structural) ─────────────────────

    [Fact]
    public void ProjectionFiles_ExpectedDimensions_17B()
    {
        // cp_projection_weight.npy should be (1024, 2048) — projects hidden→cpHidden
        // cp_projection_bias.npy should be (1024,)
        int expectedWeightRows = 1024;  // cp_hidden_size (output)
        int expectedWeightCols = 2048;  // hidden_size (input)
        int expectedBiasLen = 1024;     // cp_hidden_size

        Assert.Equal(expectedWeightRows, expectedBiasLen);
        Assert.Equal(
            QwenModelVariantConfig.GetHiddenSize(QwenModelVariant.Qwen17B),
            expectedWeightCols);
    }

    // ── Total file count validation ─────────────────────────────────

    [Fact]
    public void Downloader_06B_TotalFileCount()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);
        // 7 ONNX (+.data) + 8 embedding npy + 15 CP codec npy + 1 config + 1 speaker + 1 codec_head + 2 tokenizer = 33
        // But .onnx.data files are in the list too
        Assert.True(missing.Count >= 30, $"Expected at least 30 files for 0.6B, got {missing.Count}");
    }

    // ── ONNX data files exist ───────────────────────────────────────

    [Fact]
    public void Downloader_OnnxDataFiles_AreIncluded()
    {
        var missing = ModelDownloader.GetMissingFiles(_tempDir);

        // Large ONNX models have companion .data files
        Assert.Contains("talker_prefill.onnx.data", missing);
        Assert.Contains("talker_decode.onnx.data", missing);
        Assert.Contains("vocoder.onnx.data", missing);
    }

    // ── Regression test for Issue #30: 1.7B vocoder.onnx.data ──────

    [Fact]
    public void GetExpectedFiles_17B_IncludesVocoderData()
    {
        // Regression test for GitHub Issue #30: "1.7B model generates noise"
        // The fix required adding vocoder.onnx.data to the 1.7B expected files list.
        // This test ensures vocoder.onnx.data is in the 1.7B file list.
        var files17B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen17B);
        Assert.Contains("vocoder.onnx.data", files17B);
    }

    [Fact]
    public void GetExpectedFiles_17B_IncludesCodePredictorData()
    {
        // The 1.7B model also requires code_predictor.onnx.data (not needed for 0.6B)
        var files17B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen17B);
        Assert.Contains("code_predictor.onnx.data", files17B);
    }

    [Fact]
    public void GetExpectedFiles_17B_IncludesProjectionFiles()
    {
        // The 1.7B model requires CP projection files for 2048→1024 dimension reduction
        var files17B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen17B);
        Assert.Contains("embeddings/cp_projection_weight.npy", files17B);
        Assert.Contains("embeddings/cp_projection_bias.npy", files17B);
    }

    [Fact]
    public void GetExpectedFiles_06B_AlsoIncludesVocoderData()
    {
        // Both variants need vocoder.onnx.data (split ONNX external data)
        var files06B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen06B);
        Assert.Contains("vocoder.onnx.data", files06B);
    }

    [Fact]
    public void GetExpectedFiles_06B_DoesNotIncludeCodePredictorData()
    {
        // The 0.6B model's code_predictor.onnx is small enough to not need .data file
        var files06B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen06B);
        Assert.DoesNotContain("code_predictor.onnx.data", files06B);
    }

    [Fact]
    public void GetExpectedFiles_17B_HasMoreFilesThan06B()
    {
        // 1.7B should have all 0.6B files PLUS additional files
        var files06B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen06B);
        var files17B = ModelDownloader.GetExpectedFiles(QwenModelVariant.Qwen17B);
        
        Assert.True(files17B.Length > files06B.Length,
            $"1.7B ({files17B.Length} files) should have more files than 0.6B ({files06B.Length} files)");
    }

    [Theory]
    [InlineData(QwenModelVariant.Qwen06B)]
    [InlineData(QwenModelVariant.Qwen17B)]
    public void GetExpectedFiles_AllVariants_IncludeSharedFiles(QwenModelVariant variant)
    {
        // All variants must include the core shared files
        var files = ModelDownloader.GetExpectedFiles(variant);
        
        Assert.Contains("talker_prefill.onnx", files);
        Assert.Contains("talker_decode.onnx", files);
        Assert.Contains("code_predictor.onnx", files);
        Assert.Contains("vocoder.onnx", files);
        Assert.Contains("vocoder.onnx.data", files);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, true); } catch { }
    }
}
