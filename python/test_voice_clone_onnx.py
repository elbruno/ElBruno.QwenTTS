"""
Voice cloning parity test — ONNX Python comparison.
Compares speaker embeddings from:
1. PyTorch reference (already saved by test_voice_clone_pytorch.py)
2. ONNX speaker encoder with PyTorch-compatible mel spectrogram (Python)
3. ONNX speaker encoder with C#-style mel spectrogram (simulating .NET)

This isolates whether the issue is in ONNX export or mel spectrogram computation.
"""
import os
import json
import argparse
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from librosa.filters import mel as librosa_mel_fn


def compute_mel_pytorch_style(audio: np.ndarray, sr=24000, n_fft=1024, hop_size=256,
                               win_size=1024, n_mels=128, fmin=0, fmax=12000) -> np.ndarray:
    """Compute mel spectrogram matching PyTorch reference exactly."""
    y = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Reflect padding (same as PyTorch reference)
    padding = (n_fft - hop_size) // 2  # 384
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    
    # STFT
    hann_window = torch.hann_window(win_size)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                       center=False, pad_mode="reflect", normalized=False, onesided=True,
                       return_complex=True)
    
    # Magnitude (with epsilon for numerical stability)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    
    # Mel filterbank (slaney norm via librosa)
    mel_basis = torch.from_numpy(
        librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    ).float()
    
    mel_spec = torch.matmul(mel_basis, spec)
    
    # Dynamic range compression: log(clamp(x, min=1e-5))
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    # Output shape: (1, n_mels, T) → transpose to (1, T, n_mels) for speaker encoder
    return mel_spec.transpose(1, 2).numpy()


def compute_mel_csharp_style(audio: np.ndarray, sr=24000, n_fft=1024, hop_size=256,
                              win_size=1024, n_mels=128, fmin=0, fmax=12000) -> np.ndarray:
    """Compute mel spectrogram matching FIXED C# implementation.
    
    After fix, C# now matches PyTorch:
    1. Reflect padding of (n_fft - hop_size) // 2 = 384 samples
    2. Magnitude spectrum: sqrt(r²+i²+1e-9)
    3. Slaney-normalized mel filterbank
    4. log(clamp(x, min=1e-5))
    """
    # This should now be equivalent to compute_mel_pytorch_style
    return compute_mel_pytorch_style(audio, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10))


def run_speaker_encoder(session: ort.InferenceSession, mel: np.ndarray) -> np.ndarray:
    """Run ONNX speaker encoder on mel spectrogram."""
    result = session.run(["speaker_embedding"], {"mel_spectrogram": mel.astype(np.float32)})
    return result[0].squeeze()  # (1024,)


def main():
    parser = argparse.ArgumentParser(description="ONNX parity comparison for voice cloning")
    parser.add_argument("--onnx-dir", default=r"c:\models\QwenTTSVoiceClone",
                        help="Directory with ONNX models")
    parser.add_argument("--pytorch-dir", default="python/parity_outputs/pytorch",
                        help="Directory with PyTorch reference outputs")
    parser.add_argument("--output-dir", default="python/parity_outputs/onnx",
                        help="Output directory")
    parser.add_argument("--eng-wav", default="samples/sample_voice_orig_eng.wav")
    parser.add_argument("--spa-wav", default="samples/sample_voice_orig_spa.wav")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ONNX speaker encoder
    encoder_path = os.path.join(args.onnx_dir, "speaker_encoder.onnx")
    print(f"Loading ONNX speaker encoder from {encoder_path}...")
    session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    
    # Load PyTorch reference embeddings
    pt_eng_emb = np.load(os.path.join(args.pytorch_dir, "speaker_embedding_eng.npy"))
    pt_spa_emb = np.load(os.path.join(args.pytorch_dir, "speaker_embedding_spa.npy"))
    print(f"Loaded PyTorch reference: eng norm={np.linalg.norm(pt_eng_emb):.4f}, spa norm={np.linalg.norm(pt_spa_emb):.4f}")
    
    results = {}
    
    for tag, wav_path, pt_emb in [("eng", args.eng_wav, pt_eng_emb), ("spa", args.spa_wav, pt_spa_emb)]:
        print(f"\n=== {tag.upper()} Comparison ===")
        
        # Load audio
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == 24000
        
        # 1. PyTorch-style mel → ONNX encoder
        mel_pt = compute_mel_pytorch_style(audio)
        print(f"PyTorch-style mel shape: {mel_pt.shape}")
        emb_onnx_pt_mel = run_speaker_encoder(session, mel_pt)
        
        # 2. C#-style mel → ONNX encoder
        mel_cs = compute_mel_csharp_style(audio)
        print(f"C#-style mel shape: {mel_cs.shape}")
        emb_onnx_cs_mel = run_speaker_encoder(session, mel_cs)
        
        # Save intermediates
        np.save(os.path.join(args.output_dir, f"mel_pytorch_style_{tag}.npy"), mel_pt)
        np.save(os.path.join(args.output_dir, f"mel_csharp_style_{tag}.npy"), mel_cs)
        np.save(os.path.join(args.output_dir, f"speaker_embedding_onnx_ptmel_{tag}.npy"), emb_onnx_pt_mel)
        np.save(os.path.join(args.output_dir, f"speaker_embedding_onnx_csmel_{tag}.npy"), emb_onnx_cs_mel)
        
        # Comparisons
        cos_pt_vs_onnx_ptmel = cosine_similarity(pt_emb, emb_onnx_pt_mel)
        cos_pt_vs_onnx_csmel = cosine_similarity(pt_emb, emb_onnx_cs_mel)
        cos_onnx_ptmel_vs_csmel = cosine_similarity(emb_onnx_pt_mel, emb_onnx_cs_mel)
        
        l2_pt_vs_onnx_ptmel = float(np.linalg.norm(pt_emb.flatten() - emb_onnx_pt_mel.flatten()))
        l2_pt_vs_onnx_csmel = float(np.linalg.norm(pt_emb.flatten() - emb_onnx_cs_mel.flatten()))
        
        # Mel comparison — align lengths
        min_t = min(mel_pt.shape[1], mel_cs.shape[1])
        mel_pt_aligned = mel_pt[:, :min_t, :]
        mel_cs_aligned = mel_cs[:, :min_t, :]
        mel_cos = cosine_similarity(mel_pt_aligned, mel_cs_aligned)
        mel_max_err = float(np.max(np.abs(mel_pt_aligned - mel_cs_aligned)))
        
        print(f"\n  Mel comparison (PyTorch vs C# style):")
        print(f"    Cosine similarity: {mel_cos:.6f}")
        print(f"    Shape difference: PT={mel_pt.shape} vs CS={mel_cs.shape}")
        
        print(f"\n  Speaker Embedding Comparisons:")
        print(f"    PyTorch vs ONNX(PT-mel): cosine={cos_pt_vs_onnx_ptmel:.6f}, L2={l2_pt_vs_onnx_ptmel:.4f}")
        print(f"    PyTorch vs ONNX(CS-mel): cosine={cos_pt_vs_onnx_csmel:.6f}, L2={l2_pt_vs_onnx_csmel:.4f}")
        print(f"    ONNX(PT-mel) vs ONNX(CS-mel): cosine={cos_onnx_ptmel_vs_csmel:.6f}")
        
        print(f"\n  Norms: PT={np.linalg.norm(pt_emb):.4f}, ONNX(PT-mel)={np.linalg.norm(emb_onnx_pt_mel):.4f}, ONNX(CS-mel)={np.linalg.norm(emb_onnx_cs_mel):.4f}")
        
        results[tag] = {
            "cos_pytorch_vs_onnx_ptmel": cos_pt_vs_onnx_ptmel,
            "cos_pytorch_vs_onnx_csmel": cos_pt_vs_onnx_csmel,
            "cos_onnx_ptmel_vs_csmel": cos_onnx_ptmel_vs_csmel,
            "l2_pytorch_vs_onnx_ptmel": l2_pt_vs_onnx_ptmel,
            "l2_pytorch_vs_onnx_csmel": l2_pt_vs_onnx_csmel,
            "mel_cosine_sim": mel_cos,
            "norm_pytorch": float(np.linalg.norm(pt_emb)),
            "norm_onnx_ptmel": float(np.linalg.norm(emb_onnx_pt_mel)),
            "norm_onnx_csmel": float(np.linalg.norm(emb_onnx_cs_mel)),
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("PARITY SUMMARY")
    print("=" * 60)
    for tag, r in results.items():
        print(f"\n{tag.upper()}:")
        status_onnx = "✓ PASS" if r["cos_pytorch_vs_onnx_ptmel"] > 0.99 else "✗ FAIL"
        status_csmel = "✓ PASS" if r["cos_pytorch_vs_onnx_csmel"] > 0.99 else "✗ FAIL"
        print(f"  ONNX export parity (PT mel): {status_onnx} (cosine={r['cos_pytorch_vs_onnx_ptmel']:.6f})")
        print(f"  C# mel spectrogram impact:   {status_csmel} (cosine={r['cos_pytorch_vs_onnx_csmel']:.6f})")
        print(f"  Mel spectrogram similarity:   cosine={r['mel_cosine_sim']:.6f}")
    
    # Save results
    with open(os.path.join(args.output_dir, "parity_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
