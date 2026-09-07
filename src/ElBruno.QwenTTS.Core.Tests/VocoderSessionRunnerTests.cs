using System.Reflection;
using ElBruno.QwenTTS.Models;
using Microsoft.ML.OnnxRuntime;

namespace ElBruno.QwenTTS.Core.Tests;

public class VocoderSessionRunnerTests
{
    private const string PadFailureMessage =
        "Non-zero status code returned while running Pad node. Name:'node_pad_1' " +
        "onnxruntime::Tensor::CalculateTensorStorageSize Tensor shape.Size() must be >= 0";

    [Fact]
    public void PreferredSuccess_DoesNotCreateCpuSession()
    {
        var preferred = new FakeSession();
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => preferred, () => throw new InvalidOperationException("CPU must not be created"));

        Assert.Same(preferred, runner.Run(session => session));
        Assert.Same(preferred, runner.Run(session => session));
        Assert.False(preferred.Disposed);
    }

    [Theory]
    [InlineData("node_pad_1")]
    [InlineData("another_export_pad_node")]
    public void PadShapeFailure_RetriesOnCpuAndReusesCpuForLaterCalls(string nodeName)
    {
        var preferred = new FakeSession();
        var cpu = new FakeSession();
        var cpuCreations = 0;
        var preferredCalls = 0;
        var cpuCalls = 0;
        var waveform = new float[] { 0.1f, -0.2f };
        using var runner = new VocoderSessionRunner<FakeSession>(() => preferred, () =>
        {
            cpuCreations++;
            return cpu;
        });

        float[] Decode(FakeSession session)
        {
            if (ReferenceEquals(session, preferred))
            {
                preferredCalls++;
                throw OnnxFailure(PadFailureMessage.Replace("node_pad_1", nodeName));
            }
            Assert.Same(cpu, session);
            cpuCalls++;
            return waveform;
        }

        Assert.Same(waveform, runner.Run(Decode));
        Assert.Same(waveform, runner.Run(Decode));
        Assert.Equal(1, preferredCalls);
        Assert.Equal(1, cpuCreations);
        Assert.Equal(2, cpuCalls);
        Assert.False(preferred.Disposed);
    }

    [Theory]
    [InlineData("Non-zero status code returned while running Pad node. Name:'node_pad_1' Invalid padding")]
    [InlineData("Non-zero status code returned while running Reshape node. Tensor shape.Size() must be >= 0")]
    [InlineData("CUDA out of memory")]
    public void UnrelatedOnnxFailure_IsNotRetried(string message)
    {
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => new(), () => throw new InvalidOperationException("CPU must not be created"));
        var failure = OnnxFailure(message);

        Assert.Same(failure, Assert.Throws<OnnxRuntimeException>(() =>
            runner.Run<int>(_ => throw failure)));
    }

    [Fact]
    public void NonOnnxFailure_IsNotRetried()
    {
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => new(), () => throw new InvalidOperationException("CPU must not be created"));
        var failure = new InvalidOperationException(PadFailureMessage);

        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
            runner.Run<int>(_ => throw failure)));
    }

    [Fact]
    public void DefaultCpuWithoutFallback_PadFailureIsNotRetried()
    {
        using var runner = new VocoderSessionRunner<FakeSession>(() => new(), null);
        var failure = OnnxFailure(PadFailureMessage);
        var calls = 0;

        Assert.Same(failure, Assert.Throws<OnnxRuntimeException>(() => runner.Run<int>(_ =>
        {
            calls++;
            throw failure;
        })));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void CpuFailure_IsPropagatedWithoutAnotherRetry()
    {
        var preferred = new FakeSession();
        var cpu = new FakeSession();
        var preferredCalls = 0;
        var cpuCalls = 0;
        using var runner = new VocoderSessionRunner<FakeSession>(() => preferred, () => cpu);
        var cpuFailure = OnnxFailure(PadFailureMessage);

        int Decode(FakeSession session)
        {
            if (ReferenceEquals(session, preferred))
            {
                preferredCalls++;
                throw OnnxFailure(PadFailureMessage);
            }
            cpuCalls++;
            throw cpuFailure;
        }

        Assert.Same(cpuFailure, Assert.Throws<OnnxRuntimeException>(() => runner.Run(Decode)));
        Assert.Same(cpuFailure, Assert.Throws<OnnxRuntimeException>(() => runner.Run(Decode)));
        Assert.Equal(1, preferredCalls);
        Assert.Equal(2, cpuCalls);
    }

    [Fact]
    public void SessionInitializationFailure_IsNotRetried()
    {
        var failure = OnnxFailure(PadFailureMessage);
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => throw failure, () => throw new InvalidOperationException("CPU must not be created"));

        Assert.Same(failure, Assert.Throws<OnnxRuntimeException>(() => runner.Run(session => session)));
    }

    [Fact]
    public void CpuInitializationFailure_IsPropagatedAndNotRepeated()
    {
        var cpuCreations = 0;
        var failure = new InvalidOperationException("CPU initialization failed");
        using var runner = new VocoderSessionRunner<FakeSession>(() => new(), () =>
        {
            cpuCreations++;
            throw failure;
        });

        for (int i = 0; i < 2; i++)
            Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
                runner.Run<int>(_ => throw OnnxFailure(PadFailureMessage))));
        Assert.Equal(1, cpuCreations);
    }

    [Fact]
    public void CancelledBeforeRun_DoesNotCreateSessions()
    {
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => throw new InvalidOperationException("Preferred must not be created"),
            () => throw new InvalidOperationException("CPU must not be created"));

        Assert.Throws<OperationCanceledException>(() =>
            runner.Run(session => session, new CancellationToken(canceled: true)));
    }

    [Fact]
    public void CancellationWithPadFailure_DoesNotSwitchToCpu()
    {
        var preferred = new FakeSession();
        using var cancellation = new CancellationTokenSource();
        using var runner = new VocoderSessionRunner<FakeSession>(
            () => preferred, () => throw new InvalidOperationException("CPU must not be created"));
        var failure = OnnxFailure(PadFailureMessage);

        Assert.Same(failure, Assert.Throws<OnnxRuntimeException>(() => runner.Run<int>(_ =>
        {
            cancellation.Cancel();
            throw failure;
        }, cancellation.Token)));
        Assert.Same(preferred, runner.Run(session => session));
    }

    [Fact]
    public void CancellationDuringCpuInitialization_DoesNotRunCpu()
    {
        using var cancellation = new CancellationTokenSource();
        var preferred = new FakeSession();
        var cpu = new FakeSession();
        using var runner = new VocoderSessionRunner<FakeSession>(() => preferred, () =>
        {
            cancellation.Cancel();
            return cpu;
        });

        Assert.Throws<OperationCanceledException>(() => runner.Run<int>(session =>
        {
            Assert.Same(preferred, session);
            throw OnnxFailure(PadFailureMessage);
        }, cancellation.Token));
    }

    [Fact]
    public async Task ConcurrentPadFailures_CreateOneCpuSessionAndKeepInFlightSessionAlive()
    {
        var preferred = new FakeSession();
        var cpu = new FakeSession();
        var cpuCreations = 0;
        using var barrier = new Barrier(2);
        using var runner = new VocoderSessionRunner<FakeSession>(() => preferred, () =>
        {
            Interlocked.Increment(ref cpuCreations);
            return cpu;
        });

        FakeSession Decode(FakeSession session)
        {
            if (ReferenceEquals(session, preferred))
            {
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(10)));
                Assert.False(preferred.Disposed);
                throw OnnxFailure(PadFailureMessage);
            }
            Assert.False(preferred.Disposed);
            return session;
        }

        var results = await Task.WhenAll(
            Task.Run(() => runner.Run(Decode)),
            Task.Run(() => runner.Run(Decode)));

        Assert.All(results, result => Assert.Same(cpu, result));
        Assert.Equal(1, cpuCreations);
    }

    [Fact]
    public void Dispose_DisposesBothCreatedSessions()
    {
        var preferred = new FakeSession();
        var cpu = new FakeSession();
        var runner = new VocoderSessionRunner<FakeSession>(() => preferred, () => cpu);
        runner.Run(session => ReferenceEquals(session, cpu) ? 42 : throw OnnxFailure(PadFailureMessage));

        runner.Dispose();

        Assert.True(preferred.Disposed);
        Assert.True(cpu.Disposed);
    }

    [Fact]
    public void Dispose_DoesNotCreateUnusedSessions()
    {
        var runner = new VocoderSessionRunner<FakeSession>(
            () => throw new InvalidOperationException("Preferred must not be created"),
            () => throw new InvalidOperationException("CPU must not be created"));

        runner.Dispose();
    }

    private static OnnxRuntimeException OnnxFailure(string message)
    {
        // ORT exposes neither its exception constructor nor ErrorCode enum publicly.
        var constructor = typeof(OnnxRuntimeException)
            .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic).Single();
        var errorCode = Enum.ToObject(constructor.GetParameters()[0].ParameterType, 1);
        return (OnnxRuntimeException)constructor.Invoke([errorCode, message]);
    }

    private sealed class FakeSession : IDisposable
    {
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }
}
