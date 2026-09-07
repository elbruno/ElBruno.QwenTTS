using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace ElBruno.QwenTTS.Models;

internal sealed class VocoderSessionRunner<TSession> : IDisposable where TSession : IDisposable
{
    private readonly Lazy<TSession> _preferredSession;
    private readonly Lazy<TSession>? _cpuSession;
    private int _useCpu;

    public VocoderSessionRunner(Func<TSession> createPreferredSession, Func<TSession>? createCpuSession)
    {
        _preferredSession = new(createPreferredSession, LazyThreadSafetyMode.ExecutionAndPublication);
        _cpuSession = createCpuSession is null
            ? null
            : new(createCpuSession, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public TResult Run<TResult>(Func<TSession, TResult> run, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (Volatile.Read(ref _useCpu) != 0)
            return RunCpu(run, cancellationToken);

        // Session initialization failures are not the runtime Pad failure covered by this workaround.
        var session = _preferredSession.Value;
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            return run(session);
        }
        catch (OnnxRuntimeException ex) when (_cpuSession is not null &&
                                             !cancellationToken.IsCancellationRequested &&
                                             IsPadShapeFailure(ex))
        {
            if (Interlocked.Exchange(ref _useCpu, 1) == 0)
                Trace.TraceWarning("Vocoder Pad shape failure; retrying the vocoder on CPU. Language model execution is unchanged. {0}", ex.Message);

            return RunCpu(run, cancellationToken);
        }
    }

    private TResult RunCpu<TResult>(Func<TSession, TResult> run, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var session = _cpuSession!.Value;
        cancellationToken.ThrowIfCancellationRequested();
        return run(session);
    }

    private static bool IsPadShapeFailure(OnnxRuntimeException ex) =>
        ex.Message.Contains("while running Pad node.", StringComparison.Ordinal) &&
        ex.Message.Contains("Tensor shape.Size() must be >= 0", StringComparison.Ordinal);

    public void Dispose()
    {
        // Keep both sessions alive until disposal: concurrent runs may still be using the GPU session.
        if (_preferredSession.IsValueCreated)
            _preferredSession.Value.Dispose();
        if (_cpuSession?.IsValueCreated == true)
            _cpuSession.Value.Dispose();
    }
}
