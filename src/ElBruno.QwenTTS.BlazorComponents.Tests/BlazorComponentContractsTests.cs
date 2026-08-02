using Microsoft.AspNetCore.Components;
using ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

public class BlazorComponentContractsTests
{
    [Fact]
    public void TtsInputPanel_ExposesExpectedParameters()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.TtsInputPanel");

        var ttsService = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService");
        Assert.Equal("IQwenTtsService", ttsService.PropertyType.Name);

        var voicePreset = BlazorComponentsTestHelpers.RequireParameterProperty(type, "VoicePreset");
        Assert.Equal(typeof(string), voicePreset.PropertyType);

        var onAudioReady = BlazorComponentsTestHelpers.RequireParameterProperty(type, "OnAudioReady");
        Assert.Equal(typeof(EventCallback<Stream>), onAudioReady.PropertyType);
    }

    [Fact]
    public void VoiceSamplePlayer_ExposesExpectedParameters()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceSamplePlayer");

        var audioStream = BlazorComponentsTestHelpers.RequireParameterProperty(type, "AudioStream");
        Assert.Equal(typeof(Stream), audioStream.PropertyType);

        var label = BlazorComponentsTestHelpers.RequireParameterProperty(type, "Label");
        Assert.Equal(typeof(string), label.PropertyType);

        var autoplay = BlazorComponentsTestHelpers.RequireParameterProperty(type, "AutoPlay");
        Assert.Equal(typeof(bool), autoplay.PropertyType);
    }

    [Fact]
    public void VoiceCloningSamplePicker_ExposesExpectedParameters()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.VoiceCloningSamplePicker");

        var service = BlazorComponentsTestHelpers.RequireParameterProperty(type, "VoiceCloningService");
        Assert.Equal("IVoiceCloningService", service.PropertyType.Name);

        var onVoiceReady = BlazorComponentsTestHelpers.RequireParameterProperty(type, "OnVoiceReady");
        Assert.True(onVoiceReady.PropertyType.IsGenericType);
        Assert.Equal(typeof(EventCallback<>), onVoiceReady.PropertyType.GetGenericTypeDefinition());
        Assert.Equal("VoiceEmbedding", onVoiceReady.PropertyType.GetGenericArguments()[0].Name);
    }

    [Fact]
    public void ModelDownloadStatus_ExposesExpectedParameters()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.ModelDownloadStatus");

        var modelId = BlazorComponentsTestHelpers.RequireParameterProperty(type, "ModelId");
        Assert.Equal(typeof(string), modelId.PropertyType);

        var onModelReady = BlazorComponentsTestHelpers.RequireParameterProperty(type, "OnModelReady");
        Assert.Equal(typeof(EventCallback), onModelReady.PropertyType);
    }

    [Fact]
    public void SynthesisProgressBar_ExposesExpectedParameters()
    {
        var type = BlazorComponentsTestHelpers.RequireType("ElBruno.QwenTTS.BlazorComponents.Components.SynthesisProgressBar");

        var ttsService = BlazorComponentsTestHelpers.RequireParameterProperty(type, "TtsService");
        Assert.Equal("IQwenTtsService", ttsService.PropertyType.Name);

        var showChunkCount = BlazorComponentsTestHelpers.RequireParameterProperty(type, "ShowChunkCount");
        Assert.Equal(typeof(bool), showChunkCount.PropertyType);
    }
}
