using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Models;
using ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;
using Microsoft.AspNetCore.Components;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

public sealed class BlazorComponentCallbackTests : TestContext
{
    private object RenderByType(Type componentType, params ComponentParameter[] parameters)
    {
        var genericRender = typeof(TestContext).GetMethods()
            .First(m => m.Name == nameof(RenderComponent)
                     && m.IsGenericMethodDefinition
                     && m.GetParameters().Length == 1
                     && m.GetParameters()[0].ParameterType == typeof(ComponentParameter[]));

        return genericRender.MakeGenericMethod(componentType).Invoke(this, [parameters])!;
    }

    private static object GetInstance(object renderedComponent)
    {
        var instanceProperty = renderedComponent.GetType().GetProperty("Instance");
        Assert.NotNull(instanceProperty);
        return instanceProperty!.GetValue(renderedComponent)!;
    }

    [Fact]
    public async Task TtsInputPanel_OnAudioReady_CanBeAssignedAndInvoked()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.TtsInputPanel");
        var ttsServiceType = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService").PropertyType;
        var serviceProxy = BlazorComponentsTestHelpers.CreateNoOpProxy(ttsServiceType);

        Stream? received = null;
        var callback = EventCallback.Factory.Create<Stream>(this, stream => received = stream);
        var cut = RenderByType(type,
            ComponentParameter.CreateParameter("TtsService", serviceProxy),
            ComponentParameter.CreateParameter("OnAudioReady", callback));

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
            ComponentParameter.CreateParameter("ModelId", "qwen3-tts-0.6b"),
            ComponentParameter.CreateParameter("OnModelReady", callback));

        var instance = GetInstance(cut);
        var callbackValue = (EventCallback)type.GetProperty("OnModelReady")!.GetValue(instance)!;
        await callbackValue.InvokeAsync();

        Assert.True(invoked);
    }

    [Fact]
    public void VoiceCloneForm_RequiresTextAndReferenceAudio()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cut = (IRenderedFragment)RenderByType(type);

        cut.Find("button").Click();

        Assert.Contains("Text is required.", cut.Markup);
    }

    [Fact]
    public void VoiceCloneForm_RequiresReferenceAudio()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cut = (IRenderedFragment)RenderByType(type, ComponentParameter.CreateParameter("Text", "Hello"));

        cut.Find("button").Click();

        Assert.Contains("A reference WAV file is required.", cut.Markup);
    }

    [Fact]
    public void VoiceCloneForm_InvokesCancellationCallback()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");
        var cancelled = false;
        var callback = EventCallback.Factory.Create(this, () => cancelled = true);
        var cut = (IRenderedFragment)RenderByType(type,
            ComponentParameter.CreateParameter("IsSubmitting", true),
            ComponentParameter.CreateParameter("OnCancel", callback));

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

        var cut = (IRenderedFragment)RenderByType(formType,
            ComponentParameter.CreateParameter("Text", "Hello"),
            ComponentParameter.CreateParameter("ReferenceAudio", audio),
            ComponentParameter.CreateParameter("ReferenceTranscript", "Reference words"),
            ComponentParameter.CreateParameter("OnSubmit", callback));

        cut.Find("button").Click();

        Assert.NotNull(request);
        Assert.Equal("Reference words", request.ReferenceTranscript);
    }
}
