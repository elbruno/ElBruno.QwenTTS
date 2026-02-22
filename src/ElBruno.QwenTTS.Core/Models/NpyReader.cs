using System.Buffers.Binary;
using System.Text;

namespace ElBruno.QwenTTS.Models;

/// <summary>
/// Loads NumPy .npy files (format v1.0/v2.0) into C# arrays.
/// Supports float32 and int64 dtypes, 1D and 2D arrays only.
/// </summary>
internal static class NpyReader
{
    public static float[] ReadFloat1D(string path)
    {
        var (dtype, shape, data) = ReadNpy(path);
        if (dtype != "<f4")
            throw new InvalidDataException($"Expected float32 (<f4), got {dtype}");
        if (shape.Length != 1)
            throw new InvalidDataException($"Expected 1D array, got {shape.Length}D");

        var result = new float[shape[0]];
        Buffer.BlockCopy(data, 0, result, 0, data.Length);
        return result;
    }

    public static float[,] ReadFloat2D(string path)
    {
        var (dtype, shape, data) = ReadNpy(path);
        if (dtype != "<f4")
            throw new InvalidDataException($"Expected float32 (<f4), got {dtype}");
        if (shape.Length != 2)
            throw new InvalidDataException($"Expected 2D array, got {shape.Length}D");

        var result = new float[shape[0], shape[1]];
        Buffer.BlockCopy(data, 0, result, 0, data.Length);
        return result;
    }

    public static long[] ReadInt64_1D(string path)
    {
        var (dtype, shape, data) = ReadNpy(path);
        if (dtype != "<i8")
            throw new InvalidDataException($"Expected int64 (<i8), got {dtype}");
        if (shape.Length != 1)
            throw new InvalidDataException($"Expected 1D array, got {shape.Length}D");

        var result = new long[shape[0]];
        Buffer.BlockCopy(data, 0, result, 0, data.Length);
        return result;
    }

    private static (string dtype, int[] shape, byte[] data) ReadNpy(string path)
    {
        using var fs = File.OpenRead(path);
        
        // Read magic: 0x93 N U M P Y
        Span<byte> magic = stackalloc byte[6];
        ReadOnlySpan<byte> expected = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
        if (fs.Read(magic) != 6 || !magic.SequenceEqual(expected))
            throw new InvalidDataException("Not a valid NPY file (bad magic)");

        // Version
        int major = fs.ReadByte();
        int minor = fs.ReadByte();
        if (major is not (1 or 2))
            throw new NotSupportedException($"Unsupported NPY version {major}.{minor}");

        // Header length
        int headerLen;
        if (major == 1)
        {
            Span<byte> lenBytes = stackalloc byte[2];
            fs.Read(lenBytes);
            headerLen = BinaryPrimitives.ReadUInt16LittleEndian(lenBytes);
        }
        else // v2
        {
            Span<byte> lenBytes = stackalloc byte[4];
            fs.Read(lenBytes);
            headerLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(lenBytes);
        }

        // Read header dict (ASCII Python literal)
        var headerBytes = new byte[headerLen];
        fs.Read(headerBytes);
        var header = Encoding.ASCII.GetString(headerBytes).Trim();

        // Parse header dict {'descr': '<f4', 'fortran_order': False, 'shape': (N,M), }
        var dtype = ExtractValue(header, "'descr':");
        var shapeStr = ExtractValue(header, "'shape':");
        
        // Parse shape tuple: (N,) or (N,M) or (N,M,K)
        var shape = ParseShape(shapeStr);

        // Read data
        int elementSize = dtype switch
        {
            "<f4" => 4,
            "<i8" => 8,
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
        
        int totalElements = shape.Aggregate(1, (a, b) => a * b);
        var data = new byte[totalElements * elementSize];
        fs.Read(data);

        return (dtype, shape, data);
    }

    private static string ExtractValue(string header, string key)
    {
        var idx = header.IndexOf(key);
        if (idx < 0)
            throw new InvalidDataException($"Missing key {key} in NPY header");
        
        idx += key.Length;
        while (idx < header.Length && char.IsWhiteSpace(header[idx]))
            idx++;

        if (header[idx] == '\'')
        {
            // String value 'value'
            int start = idx + 1;
            int end = header.IndexOf('\'', start);
            return header[start..end];
        }
        else if (header[idx] == '(')
        {
            // Tuple value
            int start = idx;
            int end = header.IndexOf(')', start);
            return header[start..(end + 1)];
        }
        else
        {
            // Boolean or number
            int start = idx;
            while (idx < header.Length && !char.IsWhiteSpace(header[idx]) && header[idx] != ',')
                idx++;
            return header[start..idx];
        }
    }

    private static int[] ParseShape(string shapeStr)
    {
        // "(N,M)" or "(N,)" or "(N, M, K)"
        var inner = shapeStr.Trim('(', ')').Trim();
        var parts = inner.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Select(int.Parse).ToArray();
    }
}
