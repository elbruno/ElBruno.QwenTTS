namespace ElBruno.QwenTTS.BlazorComponents.Abstractions;

public interface IQwenTtsService
{
    Task<Stream> SynthesizeAsync(string text, string? voicePreset = null, CancellationToken cancellationToken = default);
}
