using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;
using Microsoft.AspNetCore.Components;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

public sealed class BlazorComponentRenderingTests : TestContext
{
    private IRenderedFragment RenderByType(Type componentType, params ComponentParameter[] parameters)
    {
        var genericRender = typeof(TestContext).GetMethods()
            .First(m => m.Name == nameof(RenderComponent)
                     && m.IsGenericMethodDefinition
                     && m.GetParameters().Length == 1
                     && m.GetParameters()[0].ParameterType == typeof(ComponentParameter[]));

        return (IRenderedFragment)genericRender.MakeGenericMethod(componentType).Invoke(this, [parameters])!;
    }

    [Fact]
    public void TtsInputPanel_RendersTextAreaAndButton()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.TtsInputPanel");
        var serviceType = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService").PropertyType;
        var serviceProxy = BlazorComponentsTestHelpers.CreateNoOpProxy(serviceType);

        var cut = RenderByType(type, ComponentParameter.CreateParameter("TtsService", serviceProxy));

        _ = cut.Find("textarea");
        Assert.NotEmpty(cut.FindAll("button"));
    }

    [Fact]
    public void VoiceSamplePlayer_RendersAudioTag()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceSamplePlayer");

        using var stream = new MemoryStream([1, 2, 3, 4]);
        var cut = RenderByType(type,
            ComponentParameter.CreateParameter("AudioStream", stream),
            ComponentParameter.CreateParameter("Label", "Generated voice"),
            ComponentParameter.CreateParameter("AutoPlay", false));

        _ = cut.Find("audio");
    }

    [Fact]
    public void VoiceCloningSamplePicker_RendersFileInput()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloningSamplePicker");
        var serviceType = BlazorComponentsTestHelpers.RequireParameterProperty(type, "VoiceCloningService").PropertyType;
        var serviceProxy = BlazorComponentsTestHelpers.CreateNoOpProxy(serviceType);

        var cut = RenderByType(type, ComponentParameter.CreateParameter("VoiceCloningService", serviceProxy));

        _ = cut.Find("input[type='file']");
    }

    [Fact]
    public void ModelDownloadStatus_RendersWithModelId()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.ModelDownloadStatus");

        var cut = RenderByType(type, ComponentParameter.CreateParameter("ModelId", "qwen3-tts-0.6b"));
        Assert.Contains("qwen3-tts-0.6b", cut.Markup, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SynthesisProgressBar_RendersProgressElement()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.SynthesisProgressBar");
        var serviceType = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService").PropertyType;
        var serviceProxy = BlazorComponentsTestHelpers.CreateNoOpProxy(serviceType);

        var cut = RenderByType(type,
            ComponentParameter.CreateParameter("TtsService", serviceProxy),
            ComponentParameter.CreateParameter("IsActive", true),
            ComponentParameter.CreateParameter("ShowChunkCount", true));

        Assert.True(cut.FindAll(".progress, progress, .progress-bar").Count > 0);
    }

    [Fact]
    public void VoiceCloneReferenceAudioInput_RendersWavInputWithoutRecording()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneReferenceAudioInput");

        var cut = RenderByType(type, ComponentParameter.CreateParameter("EnableRecording", false));

        var input = cut.Find("input[type='file']");
        Assert.Contains(".wav", input.GetAttribute("accept"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Maximum file size: 1 MiB.", cut.Markup);
    }

    [Fact]
    public void VoiceCloneReferenceAudioInput_ShowsUnavailableRecordingState()
    {
        JSInterop.Setup<bool>("qwenTtsRecording.isAvailable").SetResult(false);
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneReferenceAudioInput");

        var cut = RenderByType(type);

        Assert.Contains("Browser recording is unavailable.", cut.Markup);
        Assert.NotNull(cut.Find("button").GetAttribute("disabled"));
    }

    [Fact]
    public void VoiceCloneForm_RendersOptionalTranscriptAndDisabledSubmit()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneForm");

        var cut = RenderByType(type, ComponentParameter.CreateParameter("IsDisabled", true));

        Assert.Equal(2, cut.FindAll("textarea").Count);
        Assert.NotNull(cut.Find("button").GetAttribute("disabled"));
    }

    [Fact]
    public void VoiceCloneWorkflowStatus_RendersProgressMessage()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloneWorkflowStatus");
        var progressType = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Models.VoiceCloneProgress");
        var progress = Activator.CreateInstance(progressType, "Preparing reference audio", 50d, false);

        var cut = RenderByType(type, ComponentParameter.CreateParameter("Progress", progress));

        Assert.Contains("Preparing reference audio", cut.Markup);
    }
}
