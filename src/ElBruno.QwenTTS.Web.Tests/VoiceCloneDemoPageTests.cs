using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Components;
using ElBruno.QwenTTS.BlazorComponents.Models;
using ElBruno.QwenTTS.Web.Components.Pages;

namespace ElBruno.QwenTTS.Web.Tests;

public sealed class VoiceCloneDemoPageTests : TestContext
{
    [Fact]
    public async Task DemoRoute_CompletesDeterministicIclWorkflowWithoutHostServices()
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        var cut = RenderComponent<VoiceCloneDemo>();
        var reference = CreateReference();

        var input = cut.FindComponent<VoiceCloneReferenceAudioInput>();
        await input.InvokeAsync(() => input.Instance.ValueChanged.InvokeAsync(reference));

        var form = cut.FindComponent<VoiceCloneForm>();
        await form.InvokeAsync(() => form.Instance.TextChanged.InvokeAsync("Demo text"));
        await form.InvokeAsync(() => form.Instance.ReferenceTranscriptChanged.InvokeAsync("Reference words"));
        await form.InvokeAsync(() => form.Instance.OnSubmit.InvokeAsync(
            new VoiceCloneRequest("Demo text", reference, "Reference words")));

        cut.WaitForAssertion(() =>
        {
            Assert.Contains("Complete: ICL transcript demo.", cut.Markup);
            Assert.Contains("data:audio/wav;base64,", cut.Markup);
        }, timeout: TimeSpan.FromSeconds(2));
    }

    [Fact]
    public async Task DemoRoute_ReportsCancellationWithoutAccessingModels()
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        var cut = RenderComponent<VoiceCloneDemo>();
        var reference = CreateReference();
        var input = cut.FindComponent<VoiceCloneReferenceAudioInput>();
        await input.InvokeAsync(() => input.Instance.ValueChanged.InvokeAsync(reference));

        var form = cut.FindComponent<VoiceCloneForm>();
        var generation = form.InvokeAsync(() => form.Instance.OnSubmit.InvokeAsync(
            new VoiceCloneRequest("Demo text", reference, null)));
        await form.InvokeAsync(() => form.Instance.OnCancel.InvokeAsync());
        await generation;

        cut.WaitForAssertion(() =>
            Assert.Contains("Demo generation was cancelled.", cut.Markup),
            timeout: TimeSpan.FromSeconds(2));
    }

    private static VoiceCloneReferenceAudio CreateReference() =>
        new("reference.wav", "audio/wav", [0x52, 0x49, 0x46, 0x46], VoiceCloneReferenceAudioSource.Upload);
}
