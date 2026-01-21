# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "snac", "soundfile"]
# ///
"""
Decode random samples from dataset to verify SNAC codes.
Usage: uv run verify_dataset.py dataset.jsonl [-n 5]
"""

import argparse
import json
import random
from pathlib import Path
import torch
import soundfile as sf
from snac import SNAC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset.jsonl path")
    parser.add_argument("-n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    # Load samples
    samples = [json.loads(line) for line in open(args.dataset) if line.strip()]
    print(f"Loaded {len(samples)} samples")

    selected = random.sample(samples, min(args.n, len(samples)))

    # Load SNAC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    print(f"Device: {device}\n")

    for i, sample in enumerate(selected, 1):
        text = sample.get("text", "")
        codes = sample.get("snac_codes", [])
        
        if not codes:
            print(f"{i}: No snac_codes, skipping")
            continue

        print(f"{i}: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"   L1={len(codes[0])}, L2={len(codes[1])}, L3={len(codes[2])}")

        tensors = [torch.tensor([c], device=device) for c in codes]
        with torch.inference_mode():
            audio = model.decode(tensors).squeeze().cpu().numpy()

        out_path = f"sample_{i:02d}.wav"
        sf.write(out_path, audio, 24000)
        print(f"   ✓ {len(audio)/24000:.2f}s → {out_path}\n")


if __name__ == "__main__":
    main()