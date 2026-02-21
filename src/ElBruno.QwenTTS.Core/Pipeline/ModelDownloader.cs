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
    public static bool IsModelReady(string modelDir)
    {
        return ExpectedFiles.All(f =>
            File.Exists(Path.Combine(modelDir, f.Replace('/', Path.DirectorySeparatorChar))));
    }

    /// <summary>
    /// Returns the list of files that are missing from the model directory.
    /// </summary>
    public static IReadOnlyList<string> GetMissingFiles(string modelDir)
    {
        return ExpectedFiles
            .Where(f => !File.Exists(Path.Combine(modelDir, f.Replace('/', Path.DirectorySeparatorChar))))
            .ToList();
    }

    /// <summary>
    /// Downloads all missing model files from HuggingFace.
    /// Skips files that already exist locally.
    /// </summary>
    public static async Task DownloadModelAsync(
        string modelDir,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(modelDir);

        var missing = GetMissingFiles(modelDir);
        if (missing.Count == 0)
        {
            progress?.Report(new ModelDownloadProgress(0, 0, null, "All model files already present."));
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
            progress?.Report(new ModelDownloadProgress(i + 1, total, file, $"Downloading {file}..."));

            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true);
            await stream.CopyToAsync(fileStream, cancellationToken);
        }

        progress?.Report(new ModelDownloadProgress(total, total, null, $"Download complete — {total} files."));
    }

    /// <summary>
    /// Ensures model files are present, downloading them if needed.
    /// Returns the model directory path.
    /// </summary>
    public static async Task<string> EnsureModelAsync(
        string modelDir,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (!IsModelReady(modelDir))
            await DownloadModelAsync(modelDir, repoId, progress, cancellationToken);
        return modelDir;
    }
}

/// <summary>
/// Progress information for model download operations.
/// </summary>
public record ModelDownloadProgress(int Current, int Total, string? FileName, string Message)
{
    public double Percentage => Total > 0 ? Current * 100.0 / Total : 0;
}
