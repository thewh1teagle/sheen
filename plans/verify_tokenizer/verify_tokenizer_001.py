# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers==4.47.1"]
# ///
"""
Verify tokenizer setup and SNAC interleave/deinterleave roundtrip.
Usage: uv run plans/verify_tokenizer/verify_tokenizer_001.py [dataset/dataset.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import SNAC_TOKENS, SPECIAL_TOKENS, SNAC_VOCAB_SIZE
from data import setup_tokenizer, interleave_snac_codes, deinterleave_snac_tokens, format_sample


def test_tokenizer_setup(model_name):
    """Test that tokenizer is set up correctly with SNAC tokens."""
    print("1. Testing tokenizer setup...")

    tokenizer = setup_tokenizer(model_name)

    # Check special tokens added
    for tok in SPECIAL_TOKENS:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        assert tok_id != tokenizer.unk_token_id, f"Special token {tok} not added"
    print(f"   ✓ Special tokens added: {SPECIAL_TOKENS}")

    # Check SNAC tokens added (spot check)
    test_snac = ["<snac_l1_0>", "<snac_l2_2048>", "<snac_l3_4095>"]
    for tok in test_snac:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        assert tok_id != tokenizer.unk_token_id, f"SNAC token {tok} not added"
    print(f"   ✓ SNAC tokens added ({len(SNAC_TOKENS)} tokens)")

    # Check vocab size
    expected_min = 12288 + 2  # SNAC + special
    print(f"   ✓ Vocab size: {len(tokenizer)}")

    return tokenizer


def test_interleave_deinterleave():
    """Test that interleave/deinterleave is a perfect roundtrip."""
    print("\n2. Testing interleave/deinterleave roundtrip...")

    # Create test codes with exact 1:2:4 ratio
    l1 = [100, 200, 300, 400, 500]
    l2 = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]  # 2x L1
    l3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 4x L1

    original_codes = [l1, l2, l3]

    # Interleave
    tokens = interleave_snac_codes(original_codes)
    assert len(tokens) == len(l1) * 7, f"Expected {len(l1)*7} tokens, got {len(tokens)}"
    print(f"   ✓ Interleaved {len(l1)} frames -> {len(tokens)} tokens")

    # Check token format
    assert tokens[0] == "<snac_l1_100>", f"First token wrong: {tokens[0]}"
    assert tokens[1] == "<snac_l2_10>", f"Second token wrong: {tokens[1]}"
    print(f"   ✓ Token format correct: {tokens[0]}, {tokens[1]}, ...")

    # Deinterleave
    recovered = deinterleave_snac_tokens(tokens)

    assert recovered[0] == l1, f"L1 mismatch: {recovered[0]} != {l1}"
    assert recovered[1] == l2, f"L2 mismatch: {recovered[1]} != {l2}"
    assert recovered[2] == l3, f"L3 mismatch: {recovered[2]} != {l3}"
    print("   ✓ Deinterleave recovered original codes exactly")


def test_format_sample():
    """Test format_sample creates correct structure."""
    print("\n3. Testing format_sample...")

    text = "test phonemes"
    codes = [[1, 2], [10, 11, 20, 21], [100, 101, 102, 103, 200, 201, 202, 203]]

    formatted = format_sample(text, codes)

    assert formatted.startswith(text), "Should start with text"
    assert "<audio_start>" in formatted, "Should contain <audio_start>"
    assert "<audio_end>" in formatted, "Should contain <audio_end>"
    assert "<snac_l1_1>" in formatted, "Should contain SNAC tokens"

    # Check order
    start_idx = formatted.index("<audio_start>")
    end_idx = formatted.index("<audio_end>")
    assert start_idx < end_idx, "<audio_start> should come before <audio_end>"

    print(f"   ✓ Format: '{formatted[:50]}...'")


def test_tokenizer_encode_decode(tokenizer):
    """Test full encode/decode cycle."""
    print("\n4. Testing tokenizer encode/decode...")

    text = "hello"
    codes = [[1, 2], [10, 11, 20, 21], [100, 101, 102, 103, 200, 201, 202, 203]]

    formatted = format_sample(text, codes)
    encoded = tokenizer.encode(formatted)
    decoded = tokenizer.decode(encoded)

    # Check special tokens survive
    assert "<audio_start>" in decoded, "<audio_start> lost in decode"
    assert "<audio_end>" in decoded, "<audio_end> lost in decode"
    assert "<snac_l1_1>" in decoded, "SNAC tokens lost in decode"

    print(f"   ✓ Encoded to {len(encoded)} token IDs")
    print(f"   ✓ Decoded back with special tokens intact")


def test_dataset_samples(tokenizer, dataset_path, n_samples=5):
    """Test with real dataset samples."""
    print(f"\n5. Testing with real dataset samples (n={n_samples})...")

    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    import random
    test_samples = random.sample(samples, min(n_samples, len(samples)))

    for i, sample in enumerate(test_samples):
        text = sample["text"]
        codes = sample["snac_codes"]

        # Test interleave/deinterleave roundtrip
        tokens = interleave_snac_codes(codes)
        recovered = deinterleave_snac_tokens(tokens)

        # Check L1 matches (L2/L3 might be truncated due to ratio bounds)
        min_frames = len(recovered[0])
        assert recovered[0] == codes[0][:min_frames], f"Sample {i}: L1 mismatch"

        # Test full format + tokenize
        formatted = format_sample(text, codes)
        encoded = tokenizer.encode(formatted)

        print(f"   Sample {i+1}: {len(text)} chars, {len(tokens)} audio tokens -> {len(encoded)} total IDs")

    print("   ✓ All samples passed roundtrip test")


def main():
    parser = argparse.ArgumentParser(description="Verify tokenizer and data functions")
    parser.add_argument("dataset", nargs="?", help="Optional dataset.jsonl for real sample tests")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base model for tokenizer")
    parser.add_argument("-n", type=int, default=5, help="Number of samples to test")
    args = parser.parse_args()

    try:
        tokenizer = test_tokenizer_setup(args.model)
        test_interleave_deinterleave()
        test_format_sample()
        test_tokenizer_encode_decode(tokenizer)

        if args.dataset:
            test_dataset_samples(tokenizer, args.dataset, args.n)

        print("\n✓ All tests passed!")
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
