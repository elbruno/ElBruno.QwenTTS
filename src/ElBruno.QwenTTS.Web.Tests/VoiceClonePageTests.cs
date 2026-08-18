using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Components;
using ElBruno.QwenTTS.BlazorComponents.Models;
using ElBruno.QwenTTS.Pipeline;
using ElBruno.QwenTTS.Web.Components.Pages;
using ElBruno.QwenTTS.Web.Services;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging.Abstractions;

namespace ElBruno.QwenTTS.Web.Tests;

public sealed class VoiceClonePageTests : TestContext
{
    [Fact]
    public async Task FailedReferenceReplacement_ClearsThePreviouslySavedReference()
    {
        using var workspace = new TestWorkspace();
        var service = new VoiceClonePipelineService(
            new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["VoiceClone:ModelDir"] = Path.Combine(workspace.RootPath, "models")
                })
                .Build(),
            new TestWebHostEnvironment(workspace.RootPath, workspace.WebRootPath),
            NullLogger<VoiceClonePipelineService>.Instance,
            new NotDownloadedPipelineFactory());
        Services.AddSingleton(service);
        JSInterop.Setup<bool>("qwenTtsRecording.isAvailable").SetResult(false);

        var cut = RenderComponent<VoiceClone>();
        var input = cut.FindComponent<VoiceCloneReferenceAudioInput>();

        await input.InvokeAsync(() => input.Instance.ValueChanged.InvokeAsync(CreateReference("first.wav")));
        Assert.Contains("Saved reference audio", cut.Markup);

        Directory.Delete(Path.Combine(workspace.WebRootPath, "references"), recursive: true);
        await input.InvokeAsync(() => input.Instance.ValueChanged.InvokeAsync(CreateReference("replacement.wav")));

        Assert.DoesNotContain("Saved reference audio", cut.Markup);
        Assert.Contains("Unable to save the reference audio:", cut.Markup);
        Assert.NotNull(cut.FindComponent<VoiceCloneForm>().Find("button").GetAttribute("disabled"));
    }

    private static VoiceCloneReferenceAudio CreateReference(string fileName) =>
        new(fileName, "audio/wav", [0x52, 0x49, 0x46, 0x46], VoiceCloneReferenceAudioSource.Upload);

    private sealed class NotDownloadedPipelineFactory : IVoiceClonePipelineFactory
    {
        public bool IsModelDownloaded(string modelDirectory) => false;

        public Task<IVoiceClonePipeline> CreateAsync(
            string modelDirectory,
            IProgress<ModelDownloadProgress> downloadProgress,
            CancellationToken cancellationToken) =>
            throw new NotSupportedException();
    }

    private sealed class TestWebHostEnvironment(string contentRootPath, string webRootPath) : IWebHostEnvironment
    {
        public string ApplicationName { get; set; } = "ElBruno.QwenTTS.Web.Tests";
        public IFileProvider WebRootFileProvider { get; set; } = new NullFileProvider();
        public string WebRootPath { get; set; } = webRootPath;
        public string EnvironmentName { get; set; } = "Testing";
        public string ContentRootPath { get; set; } = contentRootPath;
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
    }

    private sealed class TestWorkspace : IDisposable
    {
        public TestWorkspace()
        {
            RootPath = Path.Combine(Environment.CurrentDirectory, "test-artifacts", Guid.NewGuid().ToString("N"));
            WebRootPath = Path.Combine(RootPath, "wwwroot");
            Directory.CreateDirectory(WebRootPath);
        }

        public string RootPath { get; }
        public string WebRootPath { get; }

        public void Dispose()
        {
            if (Directory.Exists(RootPath))
                Directory.Delete(RootPath, recursive: true);
        }
    }
}
