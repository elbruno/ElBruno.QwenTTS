using ElBruno.QwenTTS.Models;
using ElBruno.QwenTTS.Pipeline;

namespace ElBruno.QwenTTS.Core.Tests;

public class VocoderTests : IDisposable
{
    private readonly string _modelPath = Path.Combine(Path.GetTempPath(), $"vocoder_test_{Guid.NewGuid():N}.onnx");

    public VocoderTests()
    {
        // Opset 13 fixture: Shape/Gather/Mul compute [1,1,T*1920], then Expand broadcasts
        // the float-cast sum of codes. No model weights or downloads are required.
        File.WriteAllBytes(_modelPath, Convert.FromBase64String(
            "CAg61QMKFQoFY29kZXMSBXNoYXBlIgVTaGFwZQozCgVzaGFwZQoKdGltZV9pbmRleBIJdGltZXN0ZXBzIgZHYXRoZXIqCwoEYXhpcxgAoAECCjEKCXRpbWVzdGVwcwoRc2FtcGxlc19wZXJfZnJhbWUSDHNhbXBsZV9jb3VudCIDTXVsCkIKDWJhdGNoX2NoYW5uZWwKDHNhbXBsZV9jb3VudBIOd2F2ZWZvcm1fc2hhcGUiBkNvbmNhdCoLCgRheGlzGACgAQIKKAoFY29kZXMSA3N1bSIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGACgAQIKHQoDc3VtEgV2YWx1ZSIEQ2FzdCoJCgJ0bxgBoAECCikKBXZhbHVlCg53YXZlZm9ybV9zaGFwZRIId2F2ZWZvcm0iBkV4cGFuZBIMdm9jb2Rlcl90ZXN0KhMIARAHOgECQgp0aW1lX2luZGV4KhsIARAHOgKAD0IRc2FtcGxlc19wZXJfZnJhbWUqFwgCEAc6AgEBQg1iYXRjaF9jaGFubmVsWhwKBWNvZGVzEhMKEQgHEg0KAggBCgIIEAoDEgFUYiUKCHdhdmVmb3JtEhkKFwgBEhMKAggBCgIIAQoJEgdzYW1wbGVzQgQKABAN"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decode_CpuSessionPreservesDynamicOutputAndReusesFactory(bool customFactory)
    {
        var factoryCalls = 0;
        using var vocoder = new Vocoder(_modelPath, customFactory ? () =>
        {
            factoryCalls++;
            return OrtSessionHelper.CreateCpuOptions();
        } : null);

        foreach (var frames in new[] { 1, 3, 2 })
        {
            var codes = new long[1, 16, frames];
            codes[0, 0, 0] = frames;
            var waveform = vocoder.Decode(codes);

            Assert.Equal(frames * Vocoder.SamplesPerFrame, waveform.Length);
            Assert.All(waveform, sample => Assert.Equal((float)frames, sample));
        }

        Assert.Equal(customFactory ? 1 : 0, factoryCalls);
    }

    [Fact]
    public void Decode_CancelledRequestDoesNotInitializeSession()
    {
        using var vocoder = new Vocoder(_modelPath, () =>
            throw new InvalidOperationException("Session must not be created"));

        Assert.Throws<OperationCanceledException>(() =>
            vocoder.Decode(new long[1, 16, 1], new CancellationToken(canceled: true)));
    }

    public void Dispose() => File.Delete(_modelPath);
}
