# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "snac", "soundfile"]
# ///
"""
Verify vocab.json with round-trip: SNAC codes → tokens → SNAC codes → audio
Usage: uv run verify_vocab.py dataset.jsonl vocab.json
"""

import argparse
import json
from pathlib import Path
import torch
import soundfile as sf
from snac import SNAC


def interleave_snac(codes: list[list[int]]) -> list[tuple[int, int]]:
    """Convert [L1, L2, L3] to interleaved [(layer, code), ...] frames."""
    l1, l2, l3 = codes
    tokens = []
    for i in range(len(l1)):
        # Each frame: 1 L1 + 2 L2 + 4 L3
        tokens.append((1, l1[i]))
        tokens.append((2, l2[i * 2]))
        tokens.append((2, l2[i * 2 + 1]))
        tokens.append((3, l3[i * 4]))
        tokens.append((3, l3[i * 4 + 1]))
        tokens.append((3, l3[i * 4 + 2]))
        tokens.append((3, l3[i * 4 + 3]))
    return tokens


def deinterleave_snac(tokens: list[tuple[int, int]]) -> list[list[int]]:
    """Convert interleaved tokens back to [L1, L2, L3]."""
    l1, l2, l3 = [], [], []
    for layer, code in tokens:
        if layer == 1:
            l1.append(code)
        elif layer == 2:
            l2.append(code)
        else:
            l3.append(code)
    return [l1, l2, l3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset.jsonl path")
    parser.add_argument("vocab", help="vocab.json path")
    parser.add_argument("--output", "-o", default="verify_output.wav")
    parser.add_argument("--sample", "-n", type=int, default=0, help="Sample index")
    args = parser.parse_args()

    # Load vocab
    vocab = json.loads(Path(args.vocab).read_text())
    id2tok = {v: k for k, v in vocab.items()}
    print(f"Vocab: {len(vocab)} tokens")

    # Load one sample
    with open(args.dataset) as f:
        for i, line in enumerate(f):
            if i == args.sample:
                sample = json.loads(line)
                break
    
    text = sample["text"]
    codes = sample["snac_codes"]
    print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"SNAC: L1={len(codes[0])}, L2={len(codes[1])}, L3={len(codes[2])}")

    # Tokenize phonemes (char by char)
    text_ids = [vocab.get(c, vocab["<pad>"]) for c in text]
    print(f"Text tokens: {len(text_ids)}")

    # Interleave SNAC → token IDs
    snac_tokens = interleave_snac(codes)
    audio_ids = [vocab[f"<snac_l{layer}_{code}>"] for layer, code in snac_tokens]
    print(f"Audio tokens: {len(audio_ids)} ({len(audio_ids)//7} frames)")

    # Full sequence: text + <audio_start> + audio + <audio_end>
    full_seq = text_ids + [vocab["<audio_start>"]] + audio_ids + [vocab["<audio_end>"]]
    print(f"Full sequence: {len(full_seq)} tokens")

    # Round-trip: token IDs → SNAC codes
    reconstructed_tokens = []
    for tid in audio_ids:
        tok = id2tok[tid]  # e.g. "<snac_l1_123>"
        layer = int(tok[7])  # "l1" -> 1
        code = int(tok[9:-1])  # "123>"" -> 123
        reconstructed_tokens.append((layer, code))
    
    codes_rt = deinterleave_snac(reconstructed_tokens)
    
    # Verify round-trip
    assert codes == codes_rt, "Round-trip failed!"
    print("✓ Round-trip verified")

    # Decode to audio
    print("Decoding audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    
    out = Path(args.output)
    
    # Original
    tensors_orig = [torch.tensor([c], device=device) for c in codes]
    with torch.inference_mode():
        audio_orig = model.decode(tensors_orig).squeeze().cpu().numpy()
    orig_path = out.with_stem(out.stem + "_original")
    sf.write(orig_path, audio_orig, 24000)
    print(f"✓ Original: {len(audio_orig)/24000:.2f}s → {orig_path}")
    
    # Round-trip
    tensors_rt = [torch.tensor([c], device=device) for c in codes_rt]
    with torch.inference_mode():
        audio_rt = model.decode(tensors_rt).squeeze().cpu().numpy()
    rt_path = out.with_stem(out.stem + "_roundtrip")
    sf.write(rt_path, audio_rt, 24000)
    print(f"✓ Roundtrip: {len(audio_rt)/24000:.2f}s → {rt_path}")


if __name__ == "__main__":
    main()