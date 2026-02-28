using System.Net.Http.Json;
using System.Text.Json;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.VoiceCloning.Pipeline;

/// <summary>
/// Downloads Qwen3-TTS Base model ONNX files (including speaker encoder)
/// from a HuggingFace repository.
/// </summary>
public sealed class VoiceCloningDownloader
{
    public const string DefaultRepoId = "elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX";
    
    /// <summary>
    /// HuggingFace repository base URL. HTTPS scheme is hardcoded (not configurable) to enforce
    /// secure model downloads. This is a security design decision: all ONNX model files are
    /// large binaries (~5.5 GB total) that will be executed by ONNX Runtime. Using only HTTPS
    /// prevents Man-in-the-Middle (MITM) attacks that could inject malicious code into models
    /// during download. Attempting to change this to HTTP would bypass model signature validation
    /// and create a critical security vulnerability in the TTS inference pipeline.
    /// </summary>
    private const string HfResolveBase = "https://huggingface.co";

    /// <summary>
    /// Default model directory for voice cloning models.
    /// Stored alongside the CustomVoice models under the shared ElBruno folder.
    /// </summary>
    public static string DefaultModelDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                     "ElBruno", "QwenTTS-Base");

    private static readonly string[] ExpectedFiles =
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
    /// Returns true if the model directory contains all required files for voice cloning.
    /// </summary>
    public static bool IsModelDownloaded(string? modelDir = null)
    {
        modelDir ??= DefaultModelDir;
        return ExpectedFiles.All(f =>
        {
            ValidateRelativePath(f, nameof(f));
            return File.Exists(Path.Combine(modelDir, f.Replace('/', Path.DirectorySeparatorChar)));
        });
    }

    /// <summary>
    /// Downloads all required model files from HuggingFace.
    /// </summary>
    public static async Task DownloadModelAsync(
        string? modelDir = null,
        string repoId = DefaultRepoId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        modelDir ??= DefaultModelDir;
        Directory.CreateDirectory(modelDir);

        using var http = new HttpClient();
        http.DefaultRequestHeaders.UserAgent.ParseAdd("ElBruno.QwenTTS.VoiceCloning/1.0");

        int totalFiles = ExpectedFiles.Length;
        int currentFile = 0;

        foreach (var relativePath in ExpectedFiles)
        {
            ValidateRelativePath(relativePath, nameof(relativePath));
            currentFile++;
            var localPath = Path.Combine(modelDir, relativePath.Replace('/', Path.DirectorySeparatorChar));

            if (File.Exists(localPath))
            {
                progress?.Report(new ModelDownloadProgress(
                    currentFile, totalFiles, relativePath,
                    $"[{currentFile}/{totalFiles}] {relativePath} (cached)", 0, 0));
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);

            var url = $"{HfResolveBase}/{repoId}/resolve/main/{relativePath}";
            progress?.Report(new ModelDownloadProgress(
                currentFile, totalFiles, relativePath,
                $"[{currentFile}/{totalFiles}] Downloading {relativePath}...", 0, 0));

            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? 0;
            await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var fileStream = File.Create(localPath);

            var buffer = new byte[81920];
            long downloaded = 0;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                downloaded += bytesRead;

                progress?.Report(new ModelDownloadProgress(
                    currentFile, totalFiles, relativePath,
                    $"[{currentFile}/{totalFiles}] {relativePath} ({downloaded:N0}/{totalBytes:N0} bytes)",
                    downloaded, totalBytes));
            }
        }

        progress?.Report(new ModelDownloadProgress(
            totalFiles, totalFiles, "",
            "All model files downloaded.", 0, 0));
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
