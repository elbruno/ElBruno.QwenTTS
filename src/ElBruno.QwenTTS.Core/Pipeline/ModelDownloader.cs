using ElBruno.HuggingFace;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Downloads Qwen3-TTS ONNX model files from a HuggingFace repository.
/// Uses the ElBruno.HuggingFace.Downloader package for robust downloading with progress reporting.
/// </summary>
public sealed class ModelDownloader
{
    public const string DefaultRepoId = "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX";

    /// <summary>
    /// Default shared model directory root: %LOCALAPPDATA%/ElBruno/QwenTTS (Windows)
    /// or ~/.local/share/ElBruno/QwenTTS (Linux/macOS).
    /// For variant-specific directories, use <see cref="QwenModelVariantConfig.GetDefaultModelDir"/>.
    /// </summary>
    public static string DefaultModelDir =>
        DefaultPathHelper.GetDefaultCacheDirectory("ElBruno/QwenTTS");

    private static readonly string[] ExpectedFiles =
    [
        "talker_prefill.onnx",
        "talker_prefill.onnx.data",
        "talker_decode.onnx",
        "talker_decode.onnx.data",
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
        .. Enumerable.Range(0, 15).Select(i => $"embeddings/cp_codec_embedding_{i}.npy").ToArray()
    ];

    /// <summary>
    /// Additional files required only by the 0.6B variant (vocoder data file).
    /// </summary>
    private static readonly string[] Extra06BFiles =
    [
        "vocoder.onnx.data"
    ];

    /// <summary>
    /// Additional files required only by the 1.7B variant (vocoder data, CP projection weights, and code predictor data).
    /// </summary>
    private static readonly string[] Extra17BFiles =
    [
        "vocoder.onnx.data",
        "embeddings/cp_projection_weight.npy",
        "embeddings/cp_projection_bias.npy",
        "code_predictor.onnx.data"
    ];

    /// <summary>
    /// Returns the expected file list for a given model variant.
    /// Both variants include vocoder.onnx.data (split ONNX external data).
    /// The 0.6B variant has no additional files beyond the shared set.
    /// The 1.7B variant additionally includes code_predictor.onnx.data and CP projection weight files.
    /// </summary>
    public static string[] GetExpectedFiles(QwenModelVariant variant = QwenModelVariant.Qwen06B) =>
        variant switch
        {
            QwenModelVariant.Qwen17B => [.. ExpectedFiles, .. Extra17BFiles],
            _ => [.. ExpectedFiles, .. Extra06BFiles]
        };

    /// <summary>
    /// Returns true if the model directory contains all required files.
    /// </summary>
    public static bool IsModelDownloaded(string? modelDir = null, QwenModelVariant variant = QwenModelVariant.Qwen06B)
    {
        modelDir ??= DefaultModelDir;
        using var downloader = new HuggingFaceDownloader();
        return downloader.AreFilesAvailable(GetExpectedFiles(variant), modelDir);
    }

    // Keep old name as alias for backward compatibility
    /// <summary>Alias for <see cref="IsModelDownloaded"/>.</summary>
    public static bool IsModelReady(string modelDir) => IsModelDownloaded(modelDir);

    /// <summary>
    /// Returns the list of files that are missing from the model directory.
    /// </summary>
    public static IReadOnlyList<string> GetMissingFiles(string? modelDir = null, QwenModelVariant variant = QwenModelVariant.Qwen06B)
    {
        modelDir ??= DefaultModelDir;
        using var downloader = new HuggingFaceDownloader();
        return downloader.GetMissingFiles(GetExpectedFiles(variant), modelDir);
    }

    /// <summary>
    /// Downloads all missing model files from HuggingFace with byte-level progress.
    /// Skips files that already exist locally.
    /// </summary>
    public static async Task DownloadModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default,
        QwenModelVariant variant = QwenModelVariant.Qwen06B)
    {
        modelDir ??= DefaultModelDir;
        Directory.CreateDirectory(modelDir);

        var files = GetExpectedFiles(variant);
        using var downloader = new HuggingFaceDownloader();

        // Check if already downloaded
        if (downloader.AreFilesAvailable(files, modelDir))
        {
            progress?.Report(new ModelDownloadProgress(0, 0, null, "All model files already present.", 0, 0));
            return;
        }

        // Map progress from DownloadProgress to ModelDownloadProgress
        var downloadProgress = progress != null
            ? new Progress<DownloadProgress>(p => progress.Report(MapProgress(p)))
            : null;

        var request = new DownloadRequest
        {
            RepoId = repoId,
            LocalDirectory = modelDir,
            RequiredFiles = files,
            Progress = downloadProgress
        };

        await downloader.DownloadFilesAsync(request, cancellationToken);
    }

    /// <summary>
    /// Ensures model files are present, downloading them if needed.
    /// Returns the model directory path.
    /// </summary>
    public static async Task<string> EnsureModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default,
        QwenModelVariant variant = QwenModelVariant.Qwen06B)
    {
        modelDir ??= DefaultModelDir;
        if (!IsModelDownloaded(modelDir, variant))
            await DownloadModelAsync(modelDir, repoId, progress, cancellationToken, variant);
        return modelDir;
    }

    /// <summary>
    /// Resolves the model directory and repo ID for a given variant.
    /// If <paramref name="modelDir"/> is null, uses the variant-specific default directory.
    /// If <paramref name="repoId"/> is null, uses the variant's default repo.
    /// </summary>
    public static (string ModelDir, string RepoId) ResolveForVariant(
        QwenModelVariant variant,
        string? modelDir = null,
        string? repoId = null)
    {
        var resolvedDir = modelDir ?? QwenModelVariantConfig.GetDefaultModelDir(variant);
        var resolvedRepo = repoId ?? QwenModelVariantConfig.GetRepoId(variant);
        return (resolvedDir, resolvedRepo);
    }

    private static ModelDownloadProgress MapProgress(DownloadProgress p)
    {
        return new ModelDownloadProgress(
            CurrentFile: p.CurrentFileIndex,
            TotalFiles: p.TotalFileCount,
            FileName: p.CurrentFile,
            Message: p.Message ?? "",
            BytesDownloaded: p.BytesDownloaded,
            TotalBytes: p.TotalBytes
        );
    }
}

/// <summary>
/// Progress information for model download operations.
/// </summary>
public record ModelDownloadProgress(
    int CurrentFile, int TotalFiles, string? FileName, string Message,
    long BytesDownloaded, long TotalBytes)
{
    /// <summary>File-level percentage (0–100).</summary>
    public double FilePercentage => TotalFiles > 0 ? CurrentFile * 100.0 / TotalFiles : 0;

    /// <summary>Byte-level percentage for the current file (0–100).</summary>
    public double BytePercentage => TotalBytes > 0 ? BytesDownloaded * 100.0 / TotalBytes : 0;
}
