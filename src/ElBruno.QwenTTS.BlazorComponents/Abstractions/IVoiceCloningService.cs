using ElBruno.QwenTTS.BlazorComponents.Models;

namespace ElBruno.QwenTTS.BlazorComponents.Abstractions;

public interface IVoiceCloningService
{
    Task<VoiceEmbedding> CloneAsync(Stream audioStream, CancellationToken cancellationToken = default);
}
