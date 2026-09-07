#:sdk Microsoft.NET.Sdk.Web
#:project ../../../src/ElBruno.QwenTTS.Core/ElBruno.QwenTTS.Core.csproj
#:property PublishAot=false

// Sidecar HTTP host for the qwen-tts-studio canvas.
//
// Keeps a single TtsPipeline warm in memory so the ~5.5 GB ONNX model is loaded
// once per extension lifetime instead of once per generation. The canvas talks
// to this process over loopback; synthesis runs as a job so the UI can poll
// progress and cancel long-running requests.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using ElBruno.QwenTTS.Pipeline;

var port = int.Parse(GetOption(args, "--port") ?? "0");
var artifactsDir = GetOption(args, "--artifacts")
    ?? Path.Combine(Path.GetTempPath(), "qwen-tts-studio");
var modelDir = ResolveModelDir(GetOption(args, "--model-dir"));

Directory.CreateDirectory(artifactsDir);

var engine = new Engine(modelDir, artifactsDir);
_ = engine.InitializeAsync();

var builder = WebApplication.CreateBuilder();
builder.Logging.ClearProviders();
builder.WebHost.UseUrls($"http://127.0.0.1:{port}");

var app = builder.Build();

app.MapGet("/api/state", () => Results.Json(engine.Snapshot(), JsonOptions.Default));

app.MapPost("/api/generate", async (GenerateRequest request) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
        return Results.Json(new { error = "Text is required." }, JsonOptions.Default, statusCode: 400);

    if (request.Text.Length > 10000)
        return Results.Json(new { error = "Text exceeds the 10,000 character limit." }, JsonOptions.Default, statusCode: 400);

    try
    {
        var job = await engine.EnqueueAsync(request);
        return Results.Json(job, JsonOptions.Default);
    }
    catch (Exception ex)
    {
        return Results.Json(new { error = ex.Message }, JsonOptions.Default, statusCode: 409);
    }
});

app.MapGet("/api/jobs/{id}", (string id) =>
    engine.TryGetJob(id, out var job)
        ? Results.Json(job, JsonOptions.Default)
        : Results.Json(new { error = "Unknown job." }, JsonOptions.Default, statusCode: 404));

app.MapPost("/api/jobs/{id}/cancel", (string id) =>
{
    engine.Cancel(id);
    return engine.TryGetJob(id, out var job)
        ? Results.Json(job, JsonOptions.Default)
        : Results.Json(new { error = "Unknown job." }, JsonOptions.Default, statusCode: 404);
});

app.MapGet("/api/history", () => Results.Json(engine.History(), JsonOptions.Default));

app.MapGet("/api/audio/{id}", (string id) =>
{
    if (!engine.TryGetAudioPath(id, out var path))
        return Results.Json(new { error = "Audio not available." }, JsonOptions.Default, statusCode: 404);

    return Results.File(path, "audio/wav", Path.GetFileName(path), enableRangeProcessing: true);
});

app.Run();

static string? GetOption(string[] args, string name)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
}

// Reuses whichever shared cache already holds a complete model set. The library
// default gained a `models/` segment, so an older cache can still sit in the
// parent directory; picking it up avoids re-downloading ~5.5 GB.
static string ResolveModelDir(string? overrideDir)
{
    if (!string.IsNullOrWhiteSpace(overrideDir))
        return Path.GetFullPath(overrideDir);

    var preferred = Path.GetFullPath(ModelDownloader.DefaultModelDir);
    var candidates = new List<string> { preferred };

    if (Path.GetDirectoryName(preferred) is { Length: > 0 } parent)
        candidates.Add(parent);

    return candidates.FirstOrDefault(ModelDownloader.IsModelReady) ?? preferred;
}

static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}

sealed record GenerateRequest(string Text, string? Speaker, string? Language);

sealed class JobView
{
    public required string Id { get; init; }
    public required string Status { get; init; }
    public required string Text { get; init; }
    public required string Speaker { get; init; }
    public required string Language { get; init; }
    public required string CreatedAt { get; init; }
    public required IReadOnlyList<string> Progress { get; init; }
    public double ElapsedSeconds { get; init; }
    public double? DurationSeconds { get; init; }
    public long? SizeBytes { get; init; }
    public string? FileName { get; init; }
    public string? Error { get; init; }
}

sealed class Job
{
    public required string Id { get; init; }
    public required string Text { get; init; }
    public required string Speaker { get; init; }
    public required string Language { get; init; }
    public DateTimeOffset CreatedAt { get; } = DateTimeOffset.Now;
    public Stopwatch Clock { get; } = Stopwatch.StartNew();
    public string Status { get; set; } = "queued";
    public List<string> Progress { get; } = [];
    public string? Error { get; set; }
    public string? FilePath { get; set; }
    public double? DurationSeconds { get; set; }
    public long? SizeBytes { get; set; }
    public CancellationTokenSource Cancellation { get; } = new();

    public JobView ToView()
    {
        lock (Progress)
        {
            return new JobView
            {
                Id = Id,
                Status = Status,
                Text = Text,
                Speaker = Speaker,
                Language = Language,
                CreatedAt = CreatedAt.ToString("o"),
                Progress = [.. Progress],
                ElapsedSeconds = Math.Round(Clock.Elapsed.TotalSeconds, 1),
                DurationSeconds = DurationSeconds,
                SizeBytes = SizeBytes,
                FileName = FilePath is null ? null : Path.GetFileName(FilePath),
                Error = Error
            };
        }
    }
}

sealed class Engine(string modelDir, string artifactsDir)
{
    private static readonly JsonSerializerOptions JsonOptions =
        new(JsonSerializerDefaults.Web) { WriteIndented = true };

    private readonly ConcurrentDictionary<string, Job> _jobs = new();
    private readonly List<string> _order = [];
    private readonly List<JobView> _restored = [];
    private readonly SemaphoreSlim _queue = new(1, 1);

    private TtsPipeline? _pipeline;
    private string _state = "loading";
    private string _message = "Starting up...";
    private string[] _speakers = [];

    public async Task InitializeAsync()
    {
        try
        {
            RestoreHistory();

            _message = "Checking model files...";
            var missing = ModelDownloader.GetMissingFiles(modelDir);
            if (missing.Count > 0)
                _message = $"Downloading {missing.Count} model files from HuggingFace...";

            var progress = new Progress<string>(m => _message = m);
            _pipeline = await TtsPipeline.CreateAsync(modelDir, progress: progress);

            _speakers = [.. _pipeline.Speakers.OrderBy(s => s, StringComparer.OrdinalIgnoreCase)];
            _state = "ready";
            _message = $"Model loaded from {modelDir}";
        }
        catch (Exception ex)
        {
            _state = "error";
            _message = ex.Message;
        }
    }

    public object Snapshot() => new
    {
        state = _state,
        message = _message,
        modelDir,
        artifactsDir,
        runtime = RuntimeInfo(),
        speakers = _speakers,
        languages = QwenLanguageCatalog.Options.Select(o => new { value = o.Value, label = o.Label }),
        activeJob = ActiveJobId()
    };

    private static object RuntimeInfo()
    {
        var libraryAssembly = typeof(TtsPipeline).Assembly;
        var libraryVersion = libraryAssembly.GetName().Version?.ToString() ?? "unknown";
        var informationalVersion = libraryAssembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
            .InformationalVersion ?? libraryVersion;

        return new
        {
            libraryVersion,
            informationalVersion,
            dotnetVersion = Environment.Version.ToString(),
            onnxRuntimeVersion = typeof(InferenceSession).Assembly.GetName().Version?.ToString() ?? "unknown",
            executionProvider = "CPU",
            gpuAcceleration = false,
            processArchitecture = RuntimeInformation.ProcessArchitecture.ToString(),
            osDescription = RuntimeInformation.OSDescription
        };
    }

    public Task<JobView> EnqueueAsync(GenerateRequest request)
    {
        if (_state != "ready" || _pipeline is null)
            throw new InvalidOperationException($"Engine is not ready ({_state}): {_message}");

        if (ActiveJobId() is { } running)
            throw new InvalidOperationException($"A generation is already running (job {running}). Cancel it or wait for it to finish.");

        var speaker = string.IsNullOrWhiteSpace(request.Speaker)
            ? _speakers.FirstOrDefault() ?? "ryan"
            : request.Speaker;

        if (_speakers.Length > 0 && !_speakers.Contains(speaker, StringComparer.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Unknown speaker '{speaker}'. Available: {string.Join(", ", _speakers)}");

        var language = string.IsNullOrWhiteSpace(request.Language) ? "auto" : request.Language;
        if (!language.Equals("auto", StringComparison.OrdinalIgnoreCase) && !QwenLanguageCatalog.IsSupported(language))
            throw new InvalidOperationException($"Unsupported language '{language}'.");

        var job = new Job
        {
            Id = $"gen-{DateTime.Now:yyyyMMdd-HHmmss}-{Random.Shared.Next(0x1000, 0xffff):x}",
            Text = request.Text.Trim(),
            Speaker = speaker,
            Language = language
        };

        _jobs[job.Id] = job;
        lock (_order) _order.Add(job.Id);

        _ = Task.Run(() => RunAsync(job));
        return Task.FromResult(job.ToView());
    }

    private async Task RunAsync(Job job)
    {
        await _queue.WaitAsync();
        try
        {
            job.Status = "running";
            job.Clock.Restart();
            Report(job, $"Synthesizing with voice '{job.Speaker}' ({job.Language})...");

            var progress = new Progress<string>(m => Report(job, m));
            var wav = await _pipeline!.SynthesizeWavAsync(
                job.Text, job.Speaker, job.Language, instruct: null,
                progress: progress, cancellationToken: job.Cancellation.Token);

            var path = Path.Combine(artifactsDir, $"{job.Id}.wav");
            await File.WriteAllBytesAsync(path, wav.ToArray(), CancellationToken.None);

            job.FilePath = path;
            job.SizeBytes = wav.Length;
            // 24 kHz, 16-bit mono PCM after the 44-byte RIFF header.
            job.DurationSeconds = Math.Round((wav.Length - 44) / 2.0 / 24000.0, 2);
            job.Status = "done";
            Report(job, $"Done in {job.Clock.Elapsed.TotalSeconds:F1}s — {job.DurationSeconds:F2}s of audio.");
            Persist(job);
        }
        catch (OperationCanceledException)
        {
            job.Status = "cancelled";
            Report(job, "Cancelled.");
        }
        catch (Exception ex)
        {
            job.Status = "error";
            job.Error = ex.Message;
            Report(job, $"Failed: {ex.Message}");
        }
        finally
        {
            job.Clock.Stop();
            _queue.Release();
        }
    }

    private static void Report(Job job, string message)
    {
        lock (job.Progress)
        {
            if (job.Progress.Count == 0 || job.Progress[^1] != message)
                job.Progress.Add(message);
        }
    }

    public void Cancel(string id)
    {
        if (_jobs.TryGetValue(id, out var job) && job.Status is "queued" or "running")
            job.Cancellation.Cancel();
    }

    public bool TryGetJob(string id, out JobView view)
    {
        if (_jobs.TryGetValue(id, out var job))
        {
            view = job.ToView();
            return true;
        }

        lock (_restored)
        {
            if (_restored.FirstOrDefault(v => v.Id == id) is { } saved)
            {
                view = saved;
                return true;
            }
        }

        view = null!;
        return false;
    }

    public bool TryGetAudioPath(string id, out string path)
    {
        if (_jobs.TryGetValue(id, out var job) && job.FilePath is { } file && File.Exists(file))
        {
            path = file;
            return true;
        }

        // Results from earlier host processes still live in the artifacts directory.
        var stored = Path.Combine(artifactsDir, $"{id}.wav");
        if (id.StartsWith("gen-", StringComparison.Ordinal) && File.Exists(stored))
        {
            path = stored;
            return true;
        }

        path = null!;
        return false;
    }

    public IReadOnlyList<JobView> History()
    {
        List<JobView> views;
        lock (_order)
        {
            views = [.. _order
                .AsEnumerable()
                .Reverse()
                .Select(id => _jobs.TryGetValue(id, out var job) ? job.ToView() : null)
                .Where(view => view is not null)
                .Select(view => view!)];
        }

        lock (_restored)
        {
            var seen = views.Select(v => v.Id).ToHashSet(StringComparer.Ordinal);
            views.AddRange(_restored.Where(v => !seen.Contains(v.Id)));
        }

        return [.. views.Take(20)];
    }

    /// <summary>Writes a finished job next to its WAV so history survives host restarts.</summary>
    private void Persist(Job job)
    {
        try
        {
            var view = job.ToView();
            File.WriteAllText(
                Path.Combine(artifactsDir, $"{job.Id}.json"),
                JsonSerializer.Serialize(view, JsonOptions));
        }
        catch
        {
            // History persistence is best-effort; never fail a generation over it.
        }
    }

    /// <summary>Loads previously generated results so the canvas can replay them.</summary>
    private void RestoreHistory()
    {
        if (!Directory.Exists(artifactsDir))
            return;

        var restored = new List<JobView>();
        foreach (var file in Directory.EnumerateFiles(artifactsDir, "gen-*.json"))
        {
            try
            {
                if (JsonSerializer.Deserialize<JobView>(File.ReadAllText(file), JsonOptions) is { } view
                    && File.Exists(Path.Combine(artifactsDir, $"{view.Id}.wav")))
                {
                    restored.Add(view);
                }
            }
            catch
            {
                // Skip unreadable entries rather than blocking startup.
            }
        }

        lock (_restored)
        {
            _restored.Clear();
            _restored.AddRange(restored.OrderByDescending(v => v.CreatedAt, StringComparer.Ordinal));
        }
    }

    private string? ActiveJobId() =>
        _jobs.Values.FirstOrDefault(j => j.Status is "queued" or "running")?.Id;
}
