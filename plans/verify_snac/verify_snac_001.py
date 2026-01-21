# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.5.1", "snac==1.2.1", "soundfile==0.13.1", "transformers==4.47.1"]
# ///
"""
Verify SNAC decode works with dataset samples and interleave/deinterleave pipeline.
Usage: uv run plans/verify_snac/verify_snac_001.py dataset/dataset.jsonl [-n 3] [--output-dir outputs]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import soundfile as sf
from snac import SNAC

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import SNAC_SAMPLE_RATE
from data import interleave_snac_codes, deinterleave_snac_tokens


def decode_snac_codes(model, codes, device):
    """Decode SNAC codes to audio waveform."""
    tensors = [torch.tensor([c], device=device) for c in codes]
    with torch.inference_mode():
        audio = model.decode(tensors)
    return audio.squeeze().cpu().numpy()


def test_direct_decode(model, sample, device, output_path):
    """Test decoding SNAC codes directly from dataset."""
    codes = sample["snac_codes"]
    audio = decode_snac_codes(model, codes, device)

    sf.write(output_path, audio, SNAC_SAMPLE_RATE)
    duration = len(audio) / SNAC_SAMPLE_RATE

    return duration


def test_roundtrip_decode(model, sample, device, output_path):
    """Test decode after interleave -> deinterleave roundtrip."""
    codes = sample["snac_codes"]

    # Interleave to token strings
    tokens = interleave_snac_codes(codes)

    # Deinterleave back to codes
    recovered_codes = deinterleave_snac_tokens(tokens)

    # Decode recovered codes
    audio = decode_snac_codes(model, recovered_codes, device)

    sf.write(output_path, audio, SNAC_SAMPLE_RATE)
    duration = len(audio) / SNAC_SAMPLE_RATE

    # Check codes match (up to what interleave processed)
    n_frames = len(recovered_codes[0])
    l1_match = recovered_codes[0] == codes[0][:n_frames]
    l2_match = recovered_codes[1] == codes[1][:n_frames * 2]
    l3_match = recovered_codes[2] == codes[2][:n_frames * 4]

    return duration, l1_match and l2_match and l3_match


def main():
    # Default output dir is in the same folder as this script
    script_dir = Path(__file__).parent
    default_output = script_dir / "outputs"

    parser = argparse.ArgumentParser(description="Verify SNAC decode with dataset samples")
    parser.add_argument("dataset", help="Path to dataset.jsonl")
    parser.add_argument("-n", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--output-dir", default=str(default_output), help="Output directory for audio files")
    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(samples)} samples\n")

    # Load SNAC model
    print("Loading SNAC model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    print(f"Device: {device}\n")

    # Test samples
    import random
    test_samples = random.sample(samples, min(args.n, len(samples)))

    all_passed = True

    for i, sample in enumerate(test_samples, 1):
        text = sample["text"]
        codes = sample["snac_codes"]

        print(f"Sample {i}: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Codes: L1={len(codes[0])}, L2={len(codes[1])}, L3={len(codes[2])}")

        # Test 1: Direct decode
        direct_path = output_dir / f"sample_{i:02d}_direct.wav"
        duration = test_direct_decode(model, sample, device, direct_path)
        print(f"  ✓ Direct decode: {duration:.2f}s -> {direct_path}")

        # Test 2: Roundtrip decode
        roundtrip_path = output_dir / f"sample_{i:02d}_roundtrip.wav"
        duration_rt, codes_match = test_roundtrip_decode(model, sample, device, roundtrip_path)

        if codes_match:
            print(f"  ✓ Roundtrip decode: {duration_rt:.2f}s -> {roundtrip_path}")
        else:
            print(f"  ✗ Roundtrip decode: codes mismatch!")
            all_passed = False

        print()

    # Summary
    if all_passed:
        print(f"✓ All {len(test_samples)} samples passed!")
        print(f"  Audio files saved to {output_dir}/")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
