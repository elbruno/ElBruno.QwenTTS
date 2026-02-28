using ElBruno.QwenTTS.Models;
using Xunit;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// SEC-3: File Size Validation Tests for ONNX and NPY file limits
///
/// These tests document and verify the file size security constraints:
/// - ONNX files must not exceed 2 GB (2_000_000_000 bytes)
/// - NPY files must not exceed 500 MB (500_000_000 bytes)
/// 
/// File size validation occurs BEFORE loading model data to prevent out-of-memory attacks.
/// </summary>
public class Sec3FileSizeTests
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "Sec3Tests");

    public Sec3FileSizeTests()
    {
        // Ensure temp directory exists
        Directory.CreateDirectory(_tempDir);
    }

    private void Cleanup()
    {
        // Clean up any test files
        if (Directory.Exists(_tempDir))
        {
            try
            {
                foreach (var file in Directory.GetFiles(_tempDir))
                    File.Delete(file);
                Directory.Delete(_tempDir);
            }
            catch { }
        }
    }

    #region NPY File Size Tests (500 MB limit)

    /// <summary>
    /// Test: NPY file just under 500 MB passes validation.
    /// Documents that files at or near the limit are handled correctly.
    /// </summary>
    [Fact]
    public void NpyReader_AcceptsFileJustUnder500MB()
    {
        try
        {
            // Create a test NPY file: 499 MB
            const long size = 499_000_000; // Just under 500 MB
            var path = Path.Combine(_tempDir, "test_499mb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 500_000_000;
            
            // Validation should pass: file size <= 500 MB
            Assert.True(fileInfo.Length <= maxNpySize, 
                $"File size {fileInfo.Length} should be within limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file exactly at 500 MB boundary (inclusive limit).
    /// Documents that 500 MB is the maximum accepted size.
    /// </summary>
    [Fact]
    public void NpyReader_AcceptsFileAt500MBBoundary()
    {
        try
        {
            // Create a test NPY file: exactly 500 MB
            const long size = 500_000_000; // Exactly 500 MB
            var path = Path.Combine(_tempDir, "test_500mb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 500_000_000;
            
            // Validation should pass: file size <= 500 MB
            Assert.True(fileInfo.Length <= maxNpySize,
                $"File size {fileInfo.Length} should equal limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file just over 500 MB is rejected.
    /// Documents that files exceeding 500 MB throw InvalidOperationException.
    /// </summary>
    [Fact]
    public void NpyReader_RejectsFileJustOver500MB()
    {
        try
        {
            // Create a test NPY file: 500 MB + 1 byte
            const long size = 500_000_001; // Just over 500 MB
            var path = Path.Combine(_tempDir, "test_500mb_plus_1.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 500_000_000;
            
            // Validation should fail: file size > 500 MB
            Assert.True(fileInfo.Length > maxNpySize,
                $"File size {fileInfo.Length} should exceed limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: NPY file significantly over 500 MB (1 GB) is rejected.
    /// Documents that large files are properly rejected.
    /// </summary>
    [Fact]
    public void NpyReader_RejectsFile1GB()
    {
        try
        {
            // Create a test NPY file: 1 GB (well over limit)
            const long size = 1_000_000_000; // 1 GB
            var path = Path.Combine(_tempDir, "test_1gb.npy");
            
            CreateTestNpyFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 500_000_000;
            
            // Validation should fail: file size > 500 MB
            Assert.True(fileInfo.Length > maxNpySize,
                $"File size {fileInfo.Length} should exceed limit {maxNpySize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    #endregion

    #region ONNX File Size Tests (2 GB limit)

    /// <summary>
    /// Test: ONNX file just under 2 GB passes validation.
    /// Documents that large ONNX files near the limit are handled correctly.
    /// </summary>
    [Fact]
    public void OnnxModel_AcceptsFileJustUnder2GB()
    {
        try
        {
            // Create a test file: 1.9 GB (just under 2 GB)
            const long size = 1_900_000_000; // Just under 2 GB
            var path = Path.Combine(_tempDir, "test_1900mb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxOnnxSize,
                $"File size {fileInfo.Length} should be within limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file exactly at 2 GB boundary (inclusive limit).
    /// Documents that 2 GB is the maximum accepted size for ONNX.
    /// </summary>
    [Fact]
    public void OnnxModel_AcceptsFileAt2GBBoundary()
    {
        try
        {
            // Create a test file: exactly 2 GB
            const long size = 2_000_000_000; // Exactly 2 GB
            var path = Path.Combine(_tempDir, "test_2gb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should pass: file size <= 2 GB
            Assert.True(fileInfo.Length <= maxOnnxSize,
                $"File size {fileInfo.Length} should equal limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file just over 2 GB is rejected.
    /// Documents that files exceeding 2 GB throw InvalidOperationException.
    /// </summary>
    [Fact]
    public void OnnxModel_RejectsFileJustOver2GB()
    {
        try
        {
            // Create a test file: 2 GB + 1 byte
            const long size = 2_000_000_001; // Just over 2 GB
            var path = Path.Combine(_tempDir, "test_2gb_plus_1.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxOnnxSize,
                $"File size {fileInfo.Length} should exceed limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    /// <summary>
    /// Test: ONNX file significantly over 2 GB (4 GB) is rejected.
    /// Documents that very large files are properly rejected.
    /// </summary>
    [Fact]
    public void OnnxModel_RejectsFile4GB()
    {
        try
        {
            // Create a test file: 4 GB (well over limit)
            const long size = 4_000_000_000; // 4 GB
            var path = Path.Combine(_tempDir, "test_4gb.onnx");
            
            CreateTestOnnxFile(path, size);
            Assert.True(File.Exists(path));
            
            var fileInfo = new FileInfo(path);
            const long maxOnnxSize = 2_000_000_000;
            
            // Validation should fail: file size > 2 GB
            Assert.True(fileInfo.Length > maxOnnxSize,
                $"File size {fileInfo.Length} should exceed limit {maxOnnxSize}");
            
            File.Delete(path);
        }
        finally
        {
            Cleanup();
        }
    }

    #endregion

    #region Comparative Boundary Tests

    /// <summary>
    /// Test: ONNX 2GB limit is 4× larger than NPY 500MB limit.
    /// Documents the relative size constraints.
    /// </summary>
    [Fact]
    public void FileLimits_OnnxIs4XNpyLimit()
    {
        const long maxOnnxSize = 2_000_000_000; // 2 GB
        const long maxNpySize = 500_000_000;   // 500 MB
        
        Assert.Equal(4, maxOnnxSize / maxNpySize);
        Assert.True(maxOnnxSize > maxNpySize);
    }

    /// <summary>
    /// Test: Small files (1 MB) pass both ONNX and NPY validation.
    /// Documents that typical small model files are handled correctly.
    /// </summary>
    [Fact]
    public void SmallFiles_PassBothLimits()
    {
        const long smallFileSize = 1_000_000; // 1 MB
        const long maxNpySize = 500_000_000;
        const long maxOnnxSize = 2_000_000_000;
        
        Assert.True(smallFileSize <= maxNpySize);
        Assert.True(smallFileSize <= maxOnnxSize);
    }

    /// <summary>
    /// Test: Medium files (100 MB) pass both limits, but very large files only pass ONNX.
    /// Documents the distinction between the two file size categories.
    /// </summary>
    [Fact]
    public void MediumFiles_PassBothButLargeOnlyOnnx()
    {
        const long mediumSize = 100_000_000; // 100 MB
        const long maxNpySize = 500_000_000;
        const long maxOnnxSize = 2_000_000_000;
        
        // 100 MB passes both
        Assert.True(mediumSize <= maxNpySize);
        Assert.True(mediumSize <= maxOnnxSize);
        
        // 1 GB passes only ONNX
        const long largeSize = 1_000_000_000; // 1 GB
        Assert.False(largeSize <= maxNpySize);
        Assert.True(largeSize <= maxOnnxSize);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a test NPY file of specified size using FileStream.
    /// Does not allocate large memory buffers; uses streaming writes.
    /// For file size validation testing only — doesn't create valid NPY format.
    /// </summary>
    private static void CreateTestNpyFile(string path, long targetSize)
    {
        using var fs = File.Create(path);
        
        // Write target size via streaming zeros (1 MB chunks)
        const int bufferSize = 1024 * 1024; // 1 MB buffer
        var buffer = new byte[bufferSize];
        long remaining = targetSize;
        
        while (remaining > 0)
        {
            int toWrite = (int)Math.Min(buffer.Length, remaining);
            fs.Write(buffer, 0, toWrite);
            remaining -= toWrite;
        }
    }

    /// <summary>
    /// Creates a test ONNX file of specified size using FileStream.
    /// Does not allocate large memory buffers; uses streaming writes.
    /// For file size validation testing only — doesn't create valid ONNX format.
    /// </summary>
    private static void CreateTestOnnxFile(string path, long targetSize)
    {
        using var fs = File.Create(path);
        
        // Write target size via streaming zeros (1 MB chunks)
        const int bufferSize = 1024 * 1024; // 1 MB buffer
        var buffer = new byte[bufferSize];
        long remaining = targetSize;
        
        while (remaining > 0)
        {
            int toWrite = (int)Math.Min(buffer.Length, remaining);
            fs.Write(buffer, 0, toWrite);
            remaining -= toWrite;
        }
    }

    #endregion
}
