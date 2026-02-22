using System.Net.Http.Json;
using System.Text.Json;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Downloads Qwen3-TTS ONNX model files from a HuggingFace repository.
/// Uses the HuggingFace Hub HTTP API — no Python or tokens needed for public repos.
/// </summary>
public sealed class ModelDownloader
{
    public const string DefaultRepoId = "elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX";
    private const string HfApiBase = "https://huggingface.co/api/models";

    /// <summary>
    /// Default shared model directory: %LOCALAPPDATA%/ElBruno/QwenTTS (Windows)
    /// or ~/.local/share/ElBruno/QwenTTS (Linux/macOS).
    /// All apps using the Core library share this location to avoid duplicate downloads.
    /// </summary>
    public static string DefaultModelDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                     "ElBruno", "QwenTTS");

    private static readonly string[] ExpectedFiles =
    [
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
    /// Returns true if the model directory contains all required files.
    /// </summary>
    public static bool IsModelDownloaded(string? modelDir = null)
    {
        modelDir ??= DefaultModelDir;
        return ExpectedFiles.All(f =>
            File.Exists(Path.Combine(modelDir, f.Replace('/', Path.DirectorySeparatorChar))));
    }

    // Keep old name as alias for backward compatibility
    /// <summary>Alias for <see cref="IsModelDownloaded"/>.</summary>
    public static bool IsModelReady(string modelDir) => IsModelDownloaded(modelDir);

    /// <summary>
    /// Returns the list of files that are missing from the model directory.
    /// </summary>
    public static IReadOnlyList<string> GetMissingFiles(string? modelDir = null)
    {
        modelDir ??= DefaultModelDir;
        return ExpectedFiles
            .Where(f => !File.Exists(Path.Combine(modelDir, f.Replace('/', Path.DirectorySeparatorChar))))
            .ToList();
    }

    /// <summary>
    /// Downloads all missing model files from HuggingFace with byte-level progress.
    /// Skips files that already exist locally.
    /// </summary>
    public static async Task DownloadModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        modelDir ??= DefaultModelDir;
        Directory.CreateDirectory(modelDir);

        var missing = GetMissingFiles(modelDir);
        if (missing.Count == 0)
        {
            progress?.Report(new ModelDownloadProgress(0, 0, null, "All model files already present.", 0, 0));
            return;
        }

        using var http = new HttpClient();
        http.DefaultRequestHeaders.UserAgent.ParseAdd("ElBruno.QwenTTS/1.0");
        http.Timeout = TimeSpan.FromMinutes(30);

        var total = missing.Count;
        for (int i = 0; i < total; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var file = missing[i];
            var localPath = Path.Combine(modelDir, file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);

            var url = $"https://huggingface.co/{repoId}/resolve/main/{file}";
            progress?.Report(new ModelDownloadProgress(i + 1, total, file, $"Downloading {file}...", 0, 0));

            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            var contentLength = response.Content.Headers.ContentLength ?? 0;
            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true);

            var buffer = new byte[81920];
            long totalBytesRead = 0;
            int bytesRead;
            while ((bytesRead = await stream.ReadAsync(buffer, cancellationToken)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                totalBytesRead += bytesRead;
                progress?.Report(new ModelDownloadProgress(
                    i + 1, total, file,
                    $"Downloading {file}... {FormatBytes(totalBytesRead)}{(contentLength > 0 ? $" / {FormatBytes(contentLength)}" : "")}",
                    totalBytesRead, contentLength));
            }
        }

        progress?.Report(new ModelDownloadProgress(total, total, null, $"Download complete — {total} files.", 0, 0));
    }

    /// <summary>
    /// Ensures model files are present, downloading them if needed.
    /// Returns the model directory path.
    /// </summary>
    public static async Task<string> EnsureModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        modelDir ??= DefaultModelDir;
        if (!IsModelDownloaded(modelDir))
            await DownloadModelAsync(modelDir, repoId, progress, cancellationToken);
        return modelDir;
    }

    private static string FormatBytes(long bytes) => bytes switch
    {
        >= 1_073_741_824 => $"{bytes / 1_073_741_824.0:F1} GB",
        >= 1_048_576 => $"{bytes / 1_048_576.0:F1} MB",
        >= 1_024 => $"{bytes / 1_024.0:F0} KB",
        _ => $"{bytes} B"
    };
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
