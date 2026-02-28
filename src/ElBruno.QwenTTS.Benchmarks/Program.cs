using BenchmarkDotNet.Running;
using ElBruno.QwenTTS.Benchmarks;

namespace ElBruno.QwenTTS.Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
    }
}
