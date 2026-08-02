using Bunit;
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
}
