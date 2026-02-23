"""
Voice cloning parity test — PyTorch reference (ground truth).
Runs Qwen3-TTS Base model voice cloning end-to-end and saves intermediates.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import soundfile as sf

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def load_model(model_dir: str, tokenizer_dir: str = None):
    """Load Qwen3-TTS Base model and speech tokenizer (vocoder)."""
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    
    print(f"Loading model from {model_dir}...")
    config = Qwen3TTSConfig.from_pretrained(model_dir)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    
    # Load speech tokenizer (vocoder) for decoding codec tokens → waveform
    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(os.path.dirname(model_dir), "Qwen3-TTS-Tokenizer-12Hz")
    
    if os.path.exists(tokenizer_dir):
        from qwen_tts.core import Qwen3TTSTokenizerV2Model
        print(f"Loading speech tokenizer from {tokenizer_dir}...")
        speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir, dtype=torch.float32)
        speech_tokenizer.eval()
        model.load_speech_tokenizer(speech_tokenizer)
        print("Speech tokenizer loaded")
    else:
        print(f"WARNING: Speech tokenizer not found at {tokenizer_dir}. Codec tokens will be saved but not decoded.")
    
    return model, config


def extract_speaker_embedding(model, wav_path: str) -> np.ndarray:
    """Extract ECAPA-TDNN speaker embedding from reference audio."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 24000:
        raise ValueError(f"Expected 24kHz audio, got {sr}Hz")
    
    print(f"Extracting speaker embedding from {os.path.basename(wav_path)} ({len(audio)/sr:.1f}s)...")
    embedding = model.extract_speaker_embedding(audio, sr=24000)
    emb_np = embedding.cpu().numpy()
    print(f"Speaker embedding shape: {emb_np.shape}, norm: {np.linalg.norm(emb_np):.4f}")
    return emb_np


def build_voice_clone_prompt(speaker_embedding: np.ndarray, batch_size: int = 1) -> dict:
    """Build the voice_clone_prompt dict expected by model.generate()."""
    emb_tensor = torch.from_numpy(speaker_embedding).float()
    if emb_tensor.dim() == 1:
        emb_tensor = emb_tensor.unsqueeze(0)  # (1024,) -> (1, 1024)
    
    return {
        "ref_spk_embedding": [emb_tensor[i] for i in range(emb_tensor.shape[0])],
        "x_vector_only_mode": [True] * batch_size,
        "icl_mode": [False] * batch_size,
        "ref_code": None,
    }


def tokenize_text(model_dir: str, text: str, language: str):
    """Tokenize text for TTS. Returns token IDs tensor."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    # TTS prompt format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids


def run_voice_clone(model, config, wav_path: str, text: str, language: str, output_dir: str, tag: str, model_dir: str):
    """Run full voice cloning pipeline and save intermediates."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract speaker embedding
    spk_emb = extract_speaker_embedding(model, wav_path)
    np.save(os.path.join(output_dir, f"speaker_embedding_{tag}.npy"), spk_emb)
    print(f"Saved speaker_embedding_{tag}.npy")
    
    # 2. Tokenize
    input_ids = tokenize_text(model_dir, text, language)
    print(f"Token IDs shape: {input_ids.shape}, IDs: {input_ids[0].tolist()}")
    np.save(os.path.join(output_dir, f"token_ids_{tag}.npy"), input_ids.numpy())
    
    # 3. Build voice clone prompt
    voice_clone_prompt = build_voice_clone_prompt(spk_emb)
    
    # 4. Run generation — returns codec tokens, NOT waveform
    print(f"Generating codec tokens for '{text}' (language={language})...")
    
    talker_codes_list, talker_hidden_states_list = model.generate(
        input_ids=[input_ids],
        languages=[language],
        speakers=[None],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    )
    
    # talker_codes_list[0] shape: (T, 16) — T timesteps, 16 codec groups
    codes = talker_codes_list[0]
    print(f"Generated codec tokens shape: {codes.shape}")
    np.save(os.path.join(output_dir, f"codec_tokens_{tag}.npy"), codes.cpu().numpy())
    
    # 5. Decode with vocoder if available
    if hasattr(model, 'speech_tokenizer') and model.speech_tokenizer is not None:
        print("Decoding with vocoder...")
        # The speech tokenizer.decode expects codes in shape (B, T, num_codebooks)
        # codes is already (T, 16), just unsqueeze for batch
        codes_for_vocoder = codes.unsqueeze(0).long()
        print(f"Vocoder input shape: {codes_for_vocoder.shape}")
        
        with torch.no_grad():
            result = model.speech_tokenizer.decode(codes_for_vocoder)
        
        # result.audio_values is a list of tensors, one per batch item
        waveform = result.audio_values[0].cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.flatten()
        
        print(f"Generated {len(waveform)} samples ({len(waveform)/24000:.2f}s)")
        
        wav_path_out = os.path.join(output_dir, f"output_{tag}.wav")
        sf.write(wav_path_out, waveform, 24000, subtype='PCM_16')
        print(f"Saved {wav_path_out}")
        np.save(os.path.join(output_dir, f"waveform_{tag}.npy"), waveform)
    else:
        print("WARNING: No speech tokenizer loaded — skipping waveform decode")
    
    return spk_emb


def main():
    parser = argparse.ArgumentParser(description="Voice cloning parity test — PyTorch reference")
    parser.add_argument("--model-dir", default="python/models/Qwen3-TTS-0.6B-Base",
                        help="Path to Qwen3-TTS Base model")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="Path to speech tokenizer (vocoder). Defaults to models/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--output-dir", default="python/parity_outputs/pytorch",
                        help="Output directory for intermediates")
    parser.add_argument("--eng-wav", default="samples/sample_voice_orig_eng.wav",
                        help="English reference audio")
    parser.add_argument("--spa-wav", default="samples/sample_voice_orig_spa.wav",
                        help="Spanish reference audio")
    args = parser.parse_args()
    
    model, config = load_model(args.model_dir, args.tokenizer_dir)
    
    # English test
    print("\n=== English Voice Clone Test ===")
    eng_emb = run_voice_clone(
        model, config,
        wav_path=args.eng_wav,
        text="This is a voice cloning test.",
        language="english",
        output_dir=args.output_dir,
        tag="eng",
        model_dir=args.model_dir,
    )
    
    # Spanish test
    print("\n=== Spanish Voice Clone Test ===")
    spa_emb = run_voice_clone(
        model, config,
        wav_path=args.spa_wav,
        text="Esta es una prueba de clonación de voz.",
        language="spanish",
        output_dir=args.output_dir,
        tag="spa",
        model_dir=args.model_dir,
    )
    
    # Cross-language embedding comparison (same person should be similar)
    cosine_sim = np.dot(eng_emb.flatten(), spa_emb.flatten()) / (
        np.linalg.norm(eng_emb) * np.linalg.norm(spa_emb)
    )
    print(f"\n=== Cross-Language Comparison ===")
    print(f"Cosine similarity (eng vs spa): {cosine_sim:.6f}")
    print(f"Expected: > 0.8 (same speaker)")
    
    # Save metadata
    meta = {
        "model_dir": args.model_dir,
        "eng_wav": args.eng_wav,
        "spa_wav": args.spa_wav,
        "eng_embedding_norm": float(np.linalg.norm(eng_emb)),
        "spa_embedding_norm": float(np.linalg.norm(spa_emb)),
        "cross_language_cosine_sim": float(cosine_sim),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
