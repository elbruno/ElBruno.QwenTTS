using Bunit;
using ElBruno.QwenTTS.BlazorComponents.Components;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

/// <summary>
/// Locks in the browser-recording JS interop contract: qwenTtsRecording.start must be invoked
/// with the timer element id so the recording duration can be rendered live.
/// </summary>
public sealed class VoiceCloneReferenceAudioInputRecordingTests : TestContext
{
    [Fact]
    public async Task StartRecording_InvokesJsInteropWithTimerElementId()
    {
        JSInterop.Setup<bool>("qwenTtsRecording.isAvailable").SetResult(true);
        var startInvocation = JSInterop.SetupVoid("qwenTtsRecording.start", invocation => invocation.Arguments.Count == 1).SetVoidResult();

        var cut = RenderComponent<VoiceCloneReferenceAudioInput>();

        // Trigger the render that resolves recording availability.
        cut.Render();

        var recordButton = cut.Find("button.btn-outline-primary");
        await recordButton.ClickAsync(new Microsoft.AspNetCore.Components.Web.MouseEventArgs());

        var invocation = Assert.Single(startInvocation.Invocations);
        var timerElementId = Assert.IsType<string>(invocation.Arguments[0]);
        Assert.False(string.IsNullOrWhiteSpace(timerElementId));

        // The timer span must exist in the DOM using the same id passed to JS,
        // otherwise the JS-side setInterval callback has nothing to update.
        var timerSpan = cut.Find($"#{timerElementId}");
        Assert.Equal("qwen-recording-timer", timerSpan.GetAttribute("class"));
    }
}
