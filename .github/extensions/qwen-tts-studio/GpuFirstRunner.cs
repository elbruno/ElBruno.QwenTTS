using Microsoft.ML.OnnxRuntime;

namespace QwenTtsStudio;

// The host serializes operations; a failed GPU pipeline is discarded before CPU retry.
internal sealed class GpuFirstRunner<T>(Func<Task<T>>? createGpu, Func<Task<T>> createCpu) : IDisposable
    where T : class, IDisposable
{
    private T? _value;

    public T Value => _value ?? throw new InvalidOperationException("Pipeline is not initialized.");
    public string ExecutionProvider { get; private set; } = "Initializing";
    public bool GpuAcceleration => ExecutionProvider == "DirectML";
    public string? FallbackReason { get; private set; }

    public async Task InitializeAsync()
    {
        if (createGpu is not null)
        {
            try
            {
                _value = await createGpu();
                ExecutionProvider = "DirectML";
                return;
            }
            catch (Exception ex) when (IsGpuFailure(ex))
            {
                FallbackReason = ex.GetBaseException().Message;
            }
        }
        else
        {
            FallbackReason = "DirectML is only available on Windows; using CPU on this platform.";
        }

        await UseCpuAsync();
    }

    public async Task<TResult> RunAsync<TResult>(
        Func<T, Task<TResult>> operation,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            return await operation(Value);
        }
        catch (Exception ex) when (GpuAcceleration && IsGpuFailure(ex) && !cancellationToken.IsCancellationRequested)
        {
            // ONNX sessions load lazily, so provider/graph failures can surface on first synthesis.
            FallbackReason = ex.GetBaseException().Message;
            progress?.Report($"GPU failed; retrying on CPU: {FallbackReason}");
            await UseCpuAsync();
            cancellationToken.ThrowIfCancellationRequested();
            return await operation(Value);
        }
    }

    private async Task UseCpuAsync()
    {
        ExecutionProvider = "Initializing";
        _value?.Dispose();
        _value = null;
        _value = await createCpu();
        ExecutionProvider = "CPU";
    }

    private static bool IsGpuFailure(Exception ex) => ex is
        OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException or
        NotSupportedException or System.Runtime.InteropServices.COMException
        || (ex is TypeInitializationException { InnerException: { } inner } && IsGpuFailure(inner));

    public void Dispose()
    {
        _value?.Dispose();
        _value = null;
    }
}
