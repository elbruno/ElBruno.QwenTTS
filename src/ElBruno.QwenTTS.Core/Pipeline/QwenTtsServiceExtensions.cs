using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.OnnxRuntime;

namespace ElBruno.QwenTTS.Pipeline;

/// <summary>
/// Extension methods for registering QwenTTS services with dependency injection.
/// </summary>
public static class QwenTtsServiceExtensions
{
    /// <summary>
    /// Registers <see cref="ITtsPipeline"/> as a singleton service using default options.
    /// Models are automatically downloaded on first use.
    /// </summary>
    public static IServiceCollection AddQwenTts(this IServiceCollection services)
        => services.AddQwenTts(_ => { });

    /// <summary>
    /// Registers <see cref="ITtsPipeline"/> as a singleton service with the specified options.
    /// Models are automatically downloaded on first use.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <param name="configure">Action to configure <see cref="QwenTtsOptions"/>.</param>
    public static IServiceCollection AddQwenTts(this IServiceCollection services, Action<QwenTtsOptions> configure)
    {
        var options = new QwenTtsOptions();
        configure(options);

        services.AddSingleton<ITtsPipeline>(sp =>
        {
            var factory = ResolveSessionOptionsFactory(options);
            var vocoderFactory = ResolveVocoderSessionOptionsFactory(options);
            var modelDir = options.ModelPath ?? ModelDownloader.DefaultModelDir;

            return TtsPipeline.CreateAsync(
                modelDir: modelDir,
                repoId: options.HuggingFaceRepo,
                sessionOptionsFactory: factory,
                vocoderSessionOptionsFactory: vocoderFactory
            ).GetAwaiter().GetResult();
        });

        return services;
    }

    internal static Func<SessionOptions>? ResolveSessionOptionsFactory(QwenTtsOptions options)
    {
        if (options.SessionOptionsFactory != null)
            return options.SessionOptionsFactory;

        return options.ExecutionProvider switch
        {
            ExecutionProvider.Cuda => () => OrtSessionHelper.CreateCudaOptions(options.GpuDeviceId),
            ExecutionProvider.DirectML => () => OrtSessionHelper.CreateDirectMlOptions(options.GpuDeviceId),
            _ => null // CPU default
        };
    }

    internal static Func<SessionOptions>? ResolveVocoderSessionOptionsFactory(QwenTtsOptions options)
    {
        if (options.SessionOptionsFactory != null)
            return null; // custom factory handles everything

        // DirectML needs CPU vocoder fallback due to unsupported ops
        return options.ExecutionProvider == ExecutionProvider.DirectML
            ? OrtSessionHelper.CreateCpuOptions
            : null;
    }
}
