using ElBruno.HuggingFace;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.VoiceCloning.Pipeline;

/// <summary>
/// Downloads Qwen3-TTS Base model ONNX files (including speaker encoder)
/// from a HuggingFace repository.
/// Uses the ElBruno.HuggingFace.Downloader package for robust downloading with progress reporting.
/// </summary>
public sealed class VoiceCloningDownloader
{
    public const string DefaultRepoId = "elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX";

    /// <summary>
    /// Default model directory for voice cloning models.
    /// Stored alongside the CustomVoice models under the shared ElBruno folder.
    /// Uses manual Path.Combine to preserve the exact path users already have cached.
    /// </summary>
    public static string DefaultModelDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                     "ElBruno", "QwenTTS-Base");

    /// <summary>
    /// Files that must be present for the model to function.
    /// </summary>
    private static readonly string[] RequiredFiles =
    [
        "speaker_encoder.onnx",
        "speaker_encoder.onnx.data",
        "talker_prefill.onnx",
        "talker_prefill.onnx.data",
        "talker_decode.onnx",
        "talker_decode.onnx.data",
        "code_predictor.onnx",
        "vocoder.onnx",
        "vocoder.onnx.data",
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
    /// Files that are part of the model but may not yet be uploaded to HuggingFace.
    /// </summary>
    private static readonly string[] OptionalFiles =
    [
        "tokenizer12hz_encode.onnx",
        "tokenizer12hz_encode.onnx.data"
    ];

    /// <summary>
    /// All expected model files (required + optional). Used for backward compatibility.
    /// </summary>
    public static string[] ExpectedFiles => [.. RequiredFiles, .. OptionalFiles];

    /// <summary>
    /// Returns true if the model directory contains all required files for voice cloning.
    /// Only checks <see cref="RequiredFiles"/> since optional files may not yet be available.
    /// </summary>
    public static bool IsModelDownloaded(string? modelDir = null)
    {
        modelDir ??= DefaultModelDir;
        using var downloader = new HuggingFaceDownloader();
        return downloader.AreFilesAvailable(RequiredFiles, modelDir);
    }

    /// <summary>
    /// Downloads all required model files from HuggingFace with byte-level progress.
    /// Skips files that already exist locally. Optional files that are not yet available
    /// on HuggingFace are handled gracefully by the downloader package.
    /// </summary>
    public static async Task DownloadModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        modelDir ??= DefaultModelDir;
        Directory.CreateDirectory(modelDir);

        using var downloader = new HuggingFaceDownloader();

        if (downloader.AreFilesAvailable(RequiredFiles, modelDir))
        {
            progress?.Report(new ModelDownloadProgress(0, 0, null, "All model files already present.", 0, 0));
            return;
        }

        var downloadProgress = progress != null
            ? new Progress<DownloadProgress>(p => progress.Report(MapProgress(p)))
            : null;

        var request = new DownloadRequest
        {
            RepoId = repoId,
            LocalDirectory = modelDir,
            RequiredFiles = RequiredFiles,
            OptionalFiles = OptionalFiles,
            Progress = downloadProgress
        };

        await downloader.DownloadFilesAsync(request, cancellationToken);
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

    /// <summary>
    /// Validates that a path is relative and does not contain path traversal sequences.
    /// Rejects absolute paths and paths containing ".." segments to prevent directory traversal attacks.
    /// </summary>
    /// <param name="relativePath">The path to validate.</param>
    /// <param name="paramName">The parameter name for error reporting.</param>
    /// <exception cref="ArgumentException">Thrown if the path is absolute or contains ".." segments.</exception>
    private static void ValidateRelativePath(string relativePath, string paramName)
    {
        if (Path.IsPathRooted(relativePath))
            throw new ArgumentException("Path must be relative.", paramName);
        if (relativePath.Contains(".."))
            throw new ArgumentException("Path traversal not allowed.", paramName);
    }
}
