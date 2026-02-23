using System;
using System.Numerics;

namespace ElBruno.QwenTTS.VoiceCloning.Audio;

/// <summary>
/// Extracts mel-spectrograms from raw audio for the ECAPA-TDNN speaker encoder.
/// Produces output compatible with Qwen3-TTS Base model expectations:
/// 24 kHz sample rate, 128 mel bins, hop length 256.
/// </summary>
public static class MelSpectrogram
{
    private const int DefaultSampleRate = 24000;
    private const int DefaultNFft = 1024;
    private const int DefaultHopLength = 256;
    private const int DefaultNMels = 128;
    private const float FMin = 0f;
    private const float FMax = 12000f; // Nyquist for 24 kHz

    /// <summary>
    /// Extract a mel-spectrogram from raw audio samples.
    /// </summary>
    /// <param name="samples">PCM float32 samples at 24 kHz, mono.</param>
    /// <param name="sampleRate">Sample rate in Hz (default 24000).</param>
    /// <param name="nFft">FFT window size (default 1024).</param>
    /// <param name="hopLength">Hop length in samples (default 256).</param>
    /// <param name="nMels">Number of mel filter banks (default 128).</param>
    /// <returns>Mel-spectrogram as float[T_mel, n_mels] — time-first layout.</returns>
    public static float[,] Extract(float[] samples, int sampleRate = DefaultSampleRate,
                                    int nFft = DefaultNFft, int hopLength = DefaultHopLength,
                                    int nMels = DefaultNMels)
    {
        // Number of frequency bins in the STFT
        int nFreqs = nFft / 2 + 1;

        // Build Hann window
        var window = BuildHannWindow(nFft);

        // Build mel filter bank (slaney norm, matching PyTorch/librosa)
        var melFilters = BuildMelFilterBank(nMels, nFreqs, sampleRate, FMin, FMax);

        // Reflect-pad the input signal (matching PyTorch: padding = (n_fft - hop_size) // 2)
        int padding = (nFft - hopLength) / 2;
        var padded = ReflectPad(samples, padding, padding);

        // Compute STFT frames on padded signal
        int numFrames = 1 + (padded.Length - nFft) / hopLength;
        if (numFrames <= 0)
            numFrames = 1;

        var melSpec = new float[numFrames, nMels];

        // Scratch buffer for FFT
        var fftBuffer = new double[nFft];
        var fftImag = new double[nFft];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopLength;

            // Apply window and copy to FFT buffer
            for (int i = 0; i < nFft; i++)
            {
                int idx = start + i;
                float sample = idx < padded.Length ? padded[idx] : 0f;
                fftBuffer[i] = sample * window[i];
                fftImag[i] = 0;
            }

            // In-place FFT
            Fft(fftBuffer, fftImag, nFft);

            // Compute magnitude spectrum (sqrt(r²+i²+1e-9)) and apply mel filters
            for (int m = 0; m < nMels; m++)
            {
                double melEnergy = 0;
                for (int k = 0; k < nFreqs; k++)
                {
                    if (melFilters[m, k] > 0)
                    {
                        double real = fftBuffer[k];
                        double imag = fftImag[k];
                        double magnitude = Math.Sqrt(real * real + imag * imag + 1e-9);
                        melEnergy += melFilters[m, k] * magnitude;
                    }
                }

                // Dynamic range compression: log(clamp(x, min=1e-5))
                melSpec[frame, m] = (float)Math.Log(Math.Max(melEnergy, 1e-5));
            }
        }

        return melSpec;
    }

    /// <summary>
    /// Read a WAV file and extract mel-spectrogram.
    /// Supports 16-bit PCM WAV at any sample rate (resamples to 24 kHz if needed).
    /// </summary>
    public static float[,] FromWavFile(string wavPath)
    {
        var (samples, sampleRate) = ReadWav(wavPath);

        if (sampleRate != DefaultSampleRate)
        {
            samples = Resample(samples, sampleRate, DefaultSampleRate);
        }

        return Extract(samples, DefaultSampleRate);
    }

    private static (float[] samples, int sampleRate) ReadWav(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // RIFF header
        var riff = new string(reader.ReadChars(4));
        if (riff != "RIFF") throw new InvalidDataException("Not a WAV file");
        reader.ReadInt32(); // file size
        var wave = new string(reader.ReadChars(4));
        if (wave != "WAVE") throw new InvalidDataException("Not a WAV file");

        // Find fmt chunk
        int sampleRate = 0;
        short bitsPerSample = 0;
        short numChannels = 0;

        while (stream.Position < stream.Length)
        {
            var chunkId = new string(reader.ReadChars(4));
            int chunkSize = reader.ReadInt32();

            if (chunkId == "fmt ")
            {
                short audioFormat = reader.ReadInt16();
                numChannels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32(); // byte rate
                reader.ReadInt16(); // block align
                bitsPerSample = reader.ReadInt16();
                if (chunkSize > 16)
                    reader.ReadBytes(chunkSize - 16);
            }
            else if (chunkId == "data")
            {
                int bytesPerSample = bitsPerSample / 8;
                int totalSamples = chunkSize / bytesPerSample;
                int monoSamples = totalSamples / numChannels;
                var samples = new float[monoSamples];

                for (int i = 0; i < monoSamples; i++)
                {
                    double sum = 0;
                    for (int ch = 0; ch < numChannels; ch++)
                    {
                        sum += bitsPerSample switch
                        {
                            16 => reader.ReadInt16() / 32768.0,
                            32 => reader.ReadSingle(),
                            _ => reader.ReadInt16() / 32768.0
                        };
                    }
                    samples[i] = (float)(sum / numChannels);
                }

                return (samples, sampleRate);
            }
            else
            {
                reader.ReadBytes(chunkSize);
            }
        }

        throw new InvalidDataException("WAV file has no data chunk");
    }

    private static float[] Resample(float[] input, int fromRate, int toRate)
    {
        double ratio = (double)toRate / fromRate;
        int outputLen = (int)(input.Length * ratio);
        var output = new float[outputLen];

        for (int i = 0; i < outputLen; i++)
        {
            double srcIdx = i / ratio;
            int idx0 = (int)srcIdx;
            int idx1 = Math.Min(idx0 + 1, input.Length - 1);
            double frac = srcIdx - idx0;
            output[i] = (float)(input[idx0] * (1 - frac) + input[idx1] * frac);
        }

        return output;
    }

    private static float[] BuildHannWindow(int size)
    {
        var window = new float[size];
        for (int i = 0; i < size; i++)
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / size));
        return window;
    }

    /// <summary>
    /// Reflect-pad an array on both sides, matching PyTorch's F.pad(mode='reflect').
    /// </summary>
    private static float[] ReflectPad(float[] input, int padLeft, int padRight)
    {
        int len = input.Length;
        if (len < 2)
        {
            // Cannot reflect-pad with less than 2 samples; zero-pad instead
            var zeroPadded = new float[padLeft + len + padRight];
            Array.Copy(input, 0, zeroPadded, padLeft, len);
            return zeroPadded;
        }

        var output = new float[padLeft + len + padRight];

        // Copy original
        Array.Copy(input, 0, output, padLeft, len);

        // Reflect index helper: maps arbitrary index to [0, len-1] via reflection
        static int ReflectIndex(int idx, int length)
        {
            if (idx < 0) idx = -idx;
            int period = 2 * (length - 1);
            if (period == 0) return 0;
            idx = idx % period;
            if (idx >= length) idx = period - idx;
            return idx;
        }

        // Left padding
        for (int i = 0; i < padLeft; i++)
            output[padLeft - 1 - i] = input[ReflectIndex(i + 1, len)];

        // Right padding
        for (int i = 0; i < padRight; i++)
            output[padLeft + len + i] = input[ReflectIndex(len - 2 - i, len)];

        return output;
    }

    private static float[,] BuildMelFilterBank(int nMels, int nFreqs, int sampleRate, float fMin, float fMax)
    {
        var filters = new float[nMels, nFreqs];

        // Convert Hz to mel scale (HTK formula, same as librosa default)
        static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
        static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        double melMin = HzToMel(fMin);
        double melMax = HzToMel(fMax);

        // Create nMels+2 equally spaced mel points
        var melPoints = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);

        // Convert mel points to Hz frequencies, then to FFT bin indices
        var hzPoints = new double[nMels + 2];
        var binIndices = new double[nMels + 2];
        double fftFreqStep = (double)sampleRate / (2.0 * (nFreqs - 1));
        for (int i = 0; i < nMels + 2; i++)
        {
            hzPoints[i] = MelToHz(melPoints[i]);
            binIndices[i] = hzPoints[i] / fftFreqStep;
        }

        // Build triangular filters with slaney normalization
        // Slaney norm: each filter is normalized by 2 / (hz_high - hz_low)
        for (int m = 0; m < nMels; m++)
        {
            double left = binIndices[m];
            double center = binIndices[m + 1];
            double right = binIndices[m + 2];

            // Slaney normalization factor: 2.0 / (hz[m+2] - hz[m])
            double enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);

            for (int k = 0; k < nFreqs; k++)
            {
                if (k >= left && k <= center && center > left)
                    filters[m, k] = (float)(enorm * (k - left) / (center - left));
                else if (k > center && k <= right && right > center)
                    filters[m, k] = (float)(enorm * (right - k) / (right - center));
            }
        }

        return filters;
    }

    /// <summary>
    /// Cooley-Tukey FFT (radix-2, in-place).
    /// </summary>
    private static void Fft(double[] real, double[] imag, int n)
    {
        // Bit-reversal permutation
        int bits = (int)Math.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        // FFT butterfly
        for (int size = 2; size <= n; size *= 2)
        {
            int half = size / 2;
            double angle = -2.0 * Math.PI / size;

            for (int i = 0; i < n; i += size)
            {
                for (int k = 0; k < half; k++)
                {
                    double cos = Math.Cos(angle * k);
                    double sin = Math.Sin(angle * k);

                    double tReal = cos * real[i + k + half] - sin * imag[i + k + half];
                    double tImag = sin * real[i + k + half] + cos * imag[i + k + half];

                    real[i + k + half] = real[i + k] - tReal;
                    imag[i + k + half] = imag[i + k] - tImag;
                    real[i + k] += tReal;
                    imag[i + k] += tImag;
                }
            }
        }
    }

    private static int BitReverse(int x, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
}
