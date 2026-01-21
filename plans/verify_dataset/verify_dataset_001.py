# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Verify dataset structure and SNAC code validity.
Usage: uv run plans/verify_dataset/verify_dataset_001.py dataset/dataset.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import SNAC_VOCAB_SIZE


def verify_sample(idx, sample):
    """Verify a single sample. Returns list of errors."""
    errors = []

    # Check required fields
    if "text" not in sample:
        errors.append("missing 'text' field")
    if "snac_codes" not in sample:
        errors.append("missing 'snac_codes' field")
        return errors

    codes = sample["snac_codes"]

    # Check structure: 3 layers
    if not isinstance(codes, list) or len(codes) != 3:
        errors.append(f"snac_codes should be 3 layers, got {len(codes) if isinstance(codes, list) else type(codes)}")
        return errors

    l1, l2, l3 = codes

    # Check layer ratios (approximately 1:2:4)
    if len(l1) > 0:
        ratio_l2 = len(l2) / len(l1)
        ratio_l3 = len(l3) / len(l1)

        if not (1.8 <= ratio_l2 <= 2.2):
            errors.append(f"L2/L1 ratio {ratio_l2:.2f} not ~2.0")
        if not (3.6 <= ratio_l3 <= 4.4):
            errors.append(f"L3/L1 ratio {ratio_l3:.2f} not ~4.0")

    # Check code ranges (0-4095)
    for layer_idx, layer in enumerate([l1, l2, l3], 1):
        for code in layer:
            if not (0 <= code < SNAC_VOCAB_SIZE):
                errors.append(f"L{layer_idx} code {code} out of range [0, {SNAC_VOCAB_SIZE})")
                break

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify dataset structure")
    parser.add_argument("dataset", help="Path to dataset.jsonl")
    parser.add_argument("--max-errors", type=int, default=10, help="Max errors to show per type")
    args = parser.parse_args()

    # Load dataset
    samples = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.dataset}\n")

    # Verify each sample
    total_errors = 0
    error_samples = []

    stats = {
        "total": len(samples),
        "valid": 0,
        "l1_lengths": [],
        "l2_lengths": [],
        "l3_lengths": [],
        "text_lengths": [],
    }

    for idx, sample in enumerate(samples):
        errors = verify_sample(idx, sample)

        if errors:
            total_errors += len(errors)
            if len(error_samples) < args.max_errors:
                error_samples.append((idx, errors))
        else:
            stats["valid"] += 1
            codes = sample["snac_codes"]
            stats["l1_lengths"].append(len(codes[0]))
            stats["l2_lengths"].append(len(codes[1]))
            stats["l3_lengths"].append(len(codes[2]))
            stats["text_lengths"].append(len(sample["text"]))

    # Report errors
    if error_samples:
        print("ERRORS:")
        for idx, errors in error_samples:
            print(f"  Sample {idx}: {', '.join(errors)}")
        if total_errors > args.max_errors:
            print(f"  ... and {total_errors - args.max_errors} more errors")
        print()

    # Report stats
    print("STATISTICS:")
    print(f"  Valid samples: {stats['valid']}/{stats['total']} ({100*stats['valid']/stats['total']:.1f}%)")

    if stats["valid"] > 0:
        avg_l1 = sum(stats["l1_lengths"]) / len(stats["l1_lengths"])
        avg_l2 = sum(stats["l2_lengths"]) / len(stats["l2_lengths"])
        avg_l3 = sum(stats["l3_lengths"]) / len(stats["l3_lengths"])
        avg_text = sum(stats["text_lengths"]) / len(stats["text_lengths"])

        print(f"  Avg L1 tokens: {avg_l1:.1f}")
        print(f"  Avg L2 tokens: {avg_l2:.1f} (ratio: {avg_l2/avg_l1:.2f})")
        print(f"  Avg L3 tokens: {avg_l3:.1f} (ratio: {avg_l3/avg_l1:.2f})")
        print(f"  Avg text length: {avg_text:.1f} chars")
        print(f"  Avg total audio tokens: {avg_l1 + avg_l2 + avg_l3:.1f}")

        # Estimate sequence lengths (7 tokens per frame + text)
        avg_frames = avg_l1  # L1 count = frame count
        avg_seq_len = avg_text + 2 + (avg_frames * 7)  # text + <audio_start>/<audio_end> + audio
        print(f"  Estimated avg sequence length: {avg_seq_len:.0f} tokens")

    # Final result
    print()
    if stats["valid"] == stats["total"]:
        print("✓ All samples valid!")
        return 0
    else:
        print(f"✗ {stats['total'] - stats['valid']} invalid samples")
        return 1


if __name__ == "__main__":
    sys.exit(main())
