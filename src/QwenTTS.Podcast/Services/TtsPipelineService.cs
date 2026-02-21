using QwenTTS.Pipeline;
using System.IO.Compression;

namespace QwenTTS.Podcast.Services;

/// <summary>
/// Singleton service that wraps TtsPipeline with thread-safe access.
/// </summary>
public sealed class TtsPipelineService : IDisposable
{
    private readonly TtsPipeline _pipeline;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly string _outputDir;

    public TtsPipelineService(IConfiguration config, IWebHostEnvironment env)
    {
        var modelDir = config["TTS:ModelDir"] ?? "python/onnx_runtime";
        // Resolve relative paths from the solution root (two levels up from web project)
        if (!Path.IsPathRooted(modelDir))
        {
            var solutionRoot = Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", ".."));
            modelDir = Path.Combine(solutionRoot, modelDir);
        }
        _pipeline = new TtsPipeline(modelDir);
        _outputDir = Path.Combine(env.WebRootPath, "generated");
        Directory.CreateDirectory(_outputDir);
    }

    public IReadOnlyCollection<string> Speakers => _pipeline.Speakers;

    /// <summary>
    /// Generates a WAV file and returns the relative URL path.
    /// </summary>
    public async Task<string> GenerateAsync(string text, string speaker, string language,
                                            string? instruct, IProgress<string>? progress = null)
    {
        var fileName = $"{Guid.NewGuid():N}.wav";
        var filePath = Path.Combine(_outputDir, fileName);

        await _semaphore.WaitAsync();
        try
        {
            await _pipeline.SynthesizeAsync(text, speaker, filePath, language, instruct, progress);
        }
        finally
        {
            _semaphore.Release();
        }

        return $"/generated/{fileName}";
    }

    /// <summary>
    /// Merges multiple WAV files (from URL paths) into a single WAV and returns the URL.
    /// </summary>
    public async Task<string> MergeWavFilesAsync(List<string> wavUrls, int silenceMs = 0)
    {
        var mergedName = $"{Guid.NewGuid():N}.wav";
        var mergedPath = Path.Combine(_outputDir, mergedName);

        await Task.Run(() =>
        {
            using var output = new FileStream(mergedPath, FileMode.Create);
            var allSamples = new List<byte>();

            // Pre-compute silence gap (16-bit PCM mono at 24kHz)
            byte[] silenceGap = [];
            if (silenceMs > 0)
                silenceGap = new byte[24000 * 2 * silenceMs / 1000];

            for (int i = 0; i < wavUrls.Count; i++)
            {
                var url = wavUrls[i];
                var fileName = url.Split('/').Last();
                var filePath = Path.Combine(_outputDir, fileName);
                if (!File.Exists(filePath)) continue;

                // Add silence between segments (not before first)
                if (i > 0 && silenceGap.Length > 0)
                    allSamples.AddRange(silenceGap);

                var bytes = File.ReadAllBytes(filePath);
                if (bytes.Length <= 44) continue;
                allSamples.AddRange(bytes.AsSpan(44).ToArray());
            }

            // Write WAV header + all PCM data
            var dataLen = allSamples.Count;
            using var writer = new BinaryWriter(output);
            writer.Write("RIFF"u8);
            writer.Write(36 + dataLen);
            writer.Write("WAVE"u8);
            writer.Write("fmt "u8);
            writer.Write(16);           // chunk size
            writer.Write((short)1);     // PCM
            writer.Write((short)1);     // mono
            writer.Write(24000);        // sample rate
            writer.Write(24000 * 2);    // byte rate
            writer.Write((short)2);     // block align
            writer.Write((short)16);    // bits per sample
            writer.Write("data"u8);
            writer.Write(dataLen);
            writer.Write(allSamples.ToArray());
        });

        return $"/generated/{mergedName}";
    }

    /// <summary>
    /// Creates a zip archive of individual WAV segments and returns the URL.
    /// </summary>
    public async Task<string?> CreateSegmentZipAsync(List<string> wavUrls, List<string> fileNames)
    {
        var validPairs = wavUrls.Zip(fileNames).Where(p => !string.IsNullOrEmpty(p.First)).ToList();
        if (validPairs.Count == 0) return null;

        var zipName = $"{Guid.NewGuid():N}.zip";
        var zipPath = Path.Combine(_outputDir, zipName);

        await Task.Run(() =>
        {
            using var zip = System.IO.Compression.ZipFile.Open(zipPath, System.IO.Compression.ZipArchiveMode.Create);
            foreach (var (url, name) in validPairs)
            {
                var srcFile = Path.Combine(_outputDir, url.Split('/').Last());
                if (File.Exists(srcFile))
                    zip.CreateEntryFromFile(srcFile, name);
            }
        });

        return $"/generated/{zipName}";
    }

    public void Dispose()
    {
        _semaphore.Dispose();
        _pipeline.Dispose();
    }
}
