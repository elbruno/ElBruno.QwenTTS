using Microsoft.Extensions.DependencyInjection;

namespace ElBruno.QwenTTS.BlazorComponents.Extensions;

/// <summary>
/// Registers services required by ElBruno.QwenTTS Blazor components.
/// </summary>
public static class QwenTtsBlazorComponentsServiceExtensions
{
    /// <summary>
    /// Adds Qwen TTS Blazor component services.
    /// </summary>
    public static IServiceCollection AddQwenTtsBlazorComponents(this IServiceCollection services)
    {
        services.AddOptions();
        services.AddSingleton<QwenTtsBlazorComponentsMarker>();
        return services;
    }

    private sealed class QwenTtsBlazorComponentsMarker;
}
