using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Models;
using ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;
using Microsoft.AspNetCore.Components;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

public sealed class BlazorComponentCallbackTests : BunitContext
{
    private IRenderedComponent<DynamicComponent> RenderByType(Type componentType, params (string Name, object? Value)[] parameters)
    {
        return Render<DynamicComponent>(p => p
            .Add(x => x.Type, componentType)
            .Add(x => x.Parameters, parameters.ToDictionary(p => p.Name, p => p.Value!)));
    }

    private static object GetInstance(IRenderedComponent<DynamicComponent> renderedComponent)
        => renderedComponent.Instance.Instance!;

    [Fact]
    public async Task TtsInputPanel_OnAudioReady_CanBeAssignedAndInvoked()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.TtsInputPanel");
        var ttsServiceType = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService").PropertyType;
        var serviceProxy = BlazorComponentsTestHelpers.CreateNoOpProxy(ttsServiceType);

        Stream? received = null;
        var callback = EventCallback.Factory.Create<Stream>(this, stream => received = stream);
        var cut = RenderByType(type,
            ("TtsService", serviceProxy),
            ("OnAudioReady", callback));

        var instance = GetInstance(cut);
        var callbackValue = (EventCallback<Stream>)type.GetProperty("OnAudioReady")!.GetValue(instance)!;
        using var stream = new MemoryStream([0x01]);
        await callbackValue.InvokeAsync(stream);

        Assert.Same(stream, received);
    }

    [Fact]
    public async Task ModelDownloadStatus_OnModelReady_CanBeAssignedAndInvoked()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.ModelDownloadStatus");
        var invoked = false;
        var callback = EventCallback.Factory.Create(this, () => invoked = true);

        var cut = RenderByType(type,
            ("ModelId", "qwen3-tts-0.6b"),
            ("OnModelReady", callback));

        var instance = GetInstance(cut);
        var callbackValue = (EventCallback)type.GetProperty("OnModelReady")!.GetValue(instance)!;
        await callbackValue.InvokeAsync();

        Assert.True(invoked);
    }

    [Fact]
    public void VoiceCloneForm_RequiresTextAndReferenceAudio()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cut = RenderByType(type);

        cut.Find("button").Click();

        Assert.Contains("Text is required.", cut.Markup);
    }

    [Fact]
    public void VoiceCloneForm_RequiresReferenceAudio()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cut = RenderByType(type, ("Text", "Hello"));

        cut.Find("button").Click();

        Assert.Contains("A reference WAV file is required.", cut.Markup);
    }

    [Fact]
    public void VoiceCloneForm_InvokesCancellationCallback()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cancelled = false;
        var callback = EventCallback.Factory.Create(this, () => cancelled = true);
        var cut = RenderByType(type,
            ("IsSubmitting", true),
            ("OnCancel", callback));

        cut.FindAll("button")[1].Click();

        Assert.True(cancelled);
    }

    [Fact]
    public void VoiceCloneForm_ForwardsOptionalTranscriptInRequest()
    {
        var formType = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var audio = new VoiceCloneReferenceAudio(
            "reference.wav",
            "audio/wav",
            [1],
            VoiceCloneReferenceAudioSource.Upload);
        VoiceCloneRequest? request = null;
        var callback = EventCallback.Factory.Create<VoiceCloneRequest>(this, value => request = value);

        var cut = RenderByType(formType,
            ("Text", "Hello"),
            ("ReferenceAudio", audio),
            ("ReferenceTranscript", "Reference words"),
            ("OnSubmit", callback));

        cut.Find("button").Click();

        Assert.NotNull(request);
        Assert.Equal("Reference words", request.ReferenceTranscript);
    }
}
