using Microsoft.ML.OnnxRuntime;
using QwenTtsStudio;

namespace ElBruno.QwenTTS.Core.Tests;

public class CanvasGpuFallbackTests
{
    [Fact]
    public async Task GpuSuccess_DoesNotCreateCpu()
    {
        var gpu = new FakePipeline();
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromResult(gpu), () => throw new InvalidOperationException("CPU must not be created"));

        Assert.Equal("Initializing", runner.ExecutionProvider);
        await runner.InitializeAsync();
        Assert.Equal(42, await runner.RunAsync(_ => Task.FromResult(42)));
        Assert.Same(gpu, runner.Value);
        Assert.Equal("DirectML", runner.ExecutionProvider);
        Assert.True(runner.GpuAcceleration);
        Assert.Null(runner.FallbackReason);
    }

    [Fact]
    public async Task NoGpuBackend_UsesCpu()
    {
        var cpu = new FakePipeline();
        using var runner = new GpuFirstRunner<FakePipeline>(null, () => Task.FromResult(cpu));

        await runner.InitializeAsync();
        Assert.Same(cpu, runner.Value);
        Assert.Equal("CPU", runner.ExecutionProvider);
        Assert.False(runner.GpuAcceleration);
        Assert.Contains("only available on Windows", runner.FallbackReason);
    }

    public static IEnumerable<object[]> GpuErrors =>
    [
        [new NotSupportedException("GPU device unavailable")],
        [new DllNotFoundException("DirectML.dll")],
        [new EntryPointNotFoundException("DML provider")],
        [new NotSupportedException("GPU unsupported")],
        [new System.Runtime.InteropServices.COMException("Device removed")],
        [new TypeInitializationException("NativeMethods", new DllNotFoundException("onnxruntime"))]
    ];

    [Theory]
    [MemberData(nameof(GpuErrors))]
    public async Task GpuInitializationFailure_UsesCpuAndReportsReason(Exception failure)
    {
        var cpu = new FakePipeline();
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromException<FakePipeline>(failure), () => Task.FromResult(cpu));

        await runner.InitializeAsync();
        Assert.Same(cpu, runner.Value);
        Assert.Equal("CPU", runner.ExecutionProvider);
        Assert.Equal(failure.GetBaseException().Message, runner.FallbackReason);
    }

    [Fact]
    public async Task LazyGpuFailure_DisposesGpuBeforeCpuRetryAndKeepsCpuForNextRequest()
    {
        var gpu = new FakePipeline();
        var cpu = new FakePipeline();
        var cpuCreations = 0;
        var gpuCalls = 0;
        var cpuCalls = 0;
        var messages = new List<string>();
        using var runner = new GpuFirstRunner<FakePipeline>(() => Task.FromResult(gpu), () =>
        {
            Assert.True(gpu.Disposed);
            cpuCreations++;
            return Task.FromResult(cpu);
        });
        await runner.InitializeAsync();

        Task<int> Synthesize(FakePipeline pipeline)
        {
            if (ReferenceEquals(pipeline, gpu))
            {
                gpuCalls++;
                throw OnnxFailure();
            }
            cpuCalls++;
            return Task.FromResult(42);
        }

        Assert.Equal(42, await runner.RunAsync(Synthesize, new InlineProgress(messages.Add)));
        Assert.Equal(42, await runner.RunAsync(Synthesize));
        Assert.Equal(1, gpuCalls);
        Assert.Equal(1, cpuCreations);
        Assert.Equal(2, cpuCalls);
        Assert.Equal("CPU", runner.ExecutionProvider);
        Assert.Contains("GPU failed; retrying on CPU", Assert.Single(messages));
    }

    [Fact]
    public async Task CpuFailure_IsNotRetried()
    {
        var calls = 0;
        using var runner = new GpuFirstRunner<FakePipeline>(null, () => Task.FromResult(new FakePipeline()));
        await runner.InitializeAsync();

        await Assert.ThrowsAsync<OnnxRuntimeException>(() => runner.RunAsync<int>(_ =>
        {
            calls++;
            throw OnnxFailure();
        }));
        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task GpuAndCpuFailure_RetriesOnlyOnceAndPropagatesCpuError()
    {
        var calls = 0;
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromResult(new FakePipeline()), () => Task.FromResult(new FakePipeline()));
        await runner.InitializeAsync();

        var failure = OnnxFailure();
        var error = await Assert.ThrowsAsync<OnnxRuntimeException>(() => runner.RunAsync<int>(_ =>
        {
            calls++;
            throw failure;
        }));
        Assert.Equal(2, calls);
        Assert.Same(failure, error);
        Assert.Equal("CPU", runner.ExecutionProvider);
    }

    [Fact]
    public async Task Cancellation_DoesNotTriggerCpuFallback()
    {
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromResult(new FakePipeline()), () => throw new InvalidOperationException("No CPU retry"));
        await runner.InitializeAsync();

        await Assert.ThrowsAsync<OperationCanceledException>(() => runner.RunAsync<int>(
            _ => throw new OperationCanceledException()));
        Assert.True(runner.GpuAcceleration);
    }

    [Fact]
    public async Task CancelledTokenWithGpuError_DoesNotTriggerCpuFallback()
    {
        using var cancellation = new CancellationTokenSource();
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromResult(new FakePipeline()), () => throw new InvalidOperationException("No CPU retry"));
        await runner.InitializeAsync();

        await Assert.ThrowsAsync<OnnxRuntimeException>(() => runner.RunAsync<int>(_ =>
        {
            cancellation.Cancel();
            throw OnnxFailure();
        }, cancellationToken: cancellation.Token));
        Assert.True(runner.GpuAcceleration);
    }

    [Fact]
    public async Task InvalidInput_DoesNotTriggerCpuFallback()
    {
        using var runner = new GpuFirstRunner<FakePipeline>(
            () => Task.FromResult(new FakePipeline()), () => throw new InvalidOperationException("No CPU retry"));
        await runner.InitializeAsync();

        await Assert.ThrowsAsync<ArgumentException>(() => runner.RunAsync<int>(
            _ => throw new ArgumentException("Unknown voice")));
        Assert.True(runner.GpuAcceleration);
    }

    private static OnnxRuntimeException OnnxFailure() =>
        Assert.Throws<OnnxRuntimeException>(() => new InferenceSession(new byte[] { 0 }));

    private sealed class FakePipeline : IDisposable
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    private sealed class InlineProgress(Action<string> report) : IProgress<string>
    {
        public void Report(string value) => report(value);
    }
}
