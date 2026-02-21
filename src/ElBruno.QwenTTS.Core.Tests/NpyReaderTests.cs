using ElBruno.QwenTTS.Models;

namespace ElBruno.QwenTTS.Core.Tests;

public class NpyReaderTests : IDisposable
{
    private readonly string _tempDir;

    public NpyReaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"qwentts_npy_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void ReadFloat1D_ValidFile_ReturnsCorrectData()
    {
        var path = Path.Combine(_tempDir, "test_1d.npy");
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        WriteNpyFloat1D(path, data);

        var result = NpyReader.ReadFloat1D(path);

        Assert.Equal(data.Length, result.Length);
        for (int i = 0; i < data.Length; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void ReadFloat2D_ValidFile_ReturnsCorrectShape()
    {
        var path = Path.Combine(_tempDir, "test_2d.npy");
        var data = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        WriteNpyFloat2D(path, data);

        var result = NpyReader.ReadFloat2D(path);

        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
        Assert.Equal(1.0f, result[0, 0]);
        Assert.Equal(6.0f, result[1, 2]);
    }

    [Fact]
    public void ReadFloat1D_NonExistentFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            NpyReader.ReadFloat1D(Path.Combine(_tempDir, "missing.npy")));
    }

    [Fact]
    public void ReadFloat1D_InvalidMagic_Throws()
    {
        var path = Path.Combine(_tempDir, "bad_magic.npy");
        File.WriteAllBytes(path, new byte[] { 0, 0, 0, 0, 0, 0, 0, 0 });

        Assert.ThrowsAny<Exception>(() => NpyReader.ReadFloat1D(path));
    }

    /// <summary>Writes a minimal NumPy v1.0 .npy file with float32 1D data.</summary>
    private static void WriteNpyFloat1D(string path, float[] data)
    {
        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({data.Length},), }}";
        WriteNpy(path, header, data.Length * 4, buf =>
        {
            Buffer.BlockCopy(data, 0, buf, 0, data.Length * 4);
        });
    }

    /// <summary>Writes a minimal NumPy v1.0 .npy file with float32 2D data.</summary>
    private static void WriteNpyFloat2D(string path, float[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}";
        WriteNpy(path, header, rows * cols * 4, buf =>
        {
            Buffer.BlockCopy(data, 0, buf, 0, rows * cols * 4);
        });
    }

    private static void WriteNpy(string path, string header, int dataSize, Action<byte[]> fillData)
    {
        // Pad header to 64-byte alignment (including magic + version + headerLen)
        var headerBytes = System.Text.Encoding.ASCII.GetBytes(header);
        int preambleLen = 10; // magic(6) + version(2) + headerLen(2)
        int totalHeaderLen = preambleLen + headerBytes.Length + 1; // +1 for newline
        int padding = (64 - totalHeaderLen % 64) % 64;
        int paddedHeaderLen = headerBytes.Length + padding + 1;

        using var fs = File.Create(path);
        // Magic: \x93NUMPY
        fs.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
        // Version 1.0
        fs.Write(new byte[] { 1, 0 });
        // Header length (little-endian uint16)
        fs.Write(BitConverter.GetBytes((ushort)paddedHeaderLen));
        // Header string
        fs.Write(headerBytes);
        // Padding spaces + newline
        for (int i = 0; i < padding; i++) fs.WriteByte((byte)' ');
        fs.WriteByte((byte)'\n');
        // Data
        var buf = new byte[dataSize];
        fillData(buf);
        fs.Write(buf);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }
}
