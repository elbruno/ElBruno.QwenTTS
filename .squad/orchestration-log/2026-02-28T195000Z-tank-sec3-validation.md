# Tank: SEC-3 File Size Validation Tests

**Date:** 2026-02-28T19:50:00Z  
**By:** Tank (Tester)  
**Task:** Write comprehensive tests for SEC-3 implementation

## Work Done

Wrote 11 new test cases in `Sec3FileSizeTests.cs`:

**NPY Tests (500 MB limit):**
- NpyReader accepts files at 499 MB, 500 MB (boundary)
- NpyReader rejects files at 500+ MB, 1 GB

**ONNX Tests (2 GB limit):**
- LanguageModel accepts files at 1.9 GB, 2 GB (boundary)
- LanguageModel rejects files at 2+ GB, 4 GB

**Comparative Tests:**
- Verify 2GB = 4× NPY limit ratio
- Document behavior across size ranges

## Test Coverage

- Before: 39 Core tests (SEC-1 validation)
- After: 50 Core tests (SEC-1 + SEC-3)
- Total: 60 tests passing (50 Core + 10 VoiceCloning)
- **Confidence:** HIGH — file size checks are deterministic

## Files Modified

- `src/ElBruno.QwenTTS.Core.Tests/Sec3FileSizeTests.cs` (NEW)

## Build Status

✅ All 50 Core tests pass
✅ All 10 VoiceCloning tests pass
✅ 0 warnings, 0 errors

## Next Step

Switch documents SEC-4 HTTPS enforcement guidance.
