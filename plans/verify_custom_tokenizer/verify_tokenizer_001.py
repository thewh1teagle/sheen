# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.21.1"]
# ///
"""
Verify custom character-level tokenizer works correctly.

Usage: uv run plans/verify_custom_tokenizer/verify_tokenizer_001.py tokenizer.json [dataset.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def test_character_level(tokenizer: Tokenizer):
    """Test that phoneme text is tokenized character-by-character."""
    print("=" * 60)
    print("1. CHARACTER-LEVEL TOKENIZATION")
    print("=" * 60)

    test_cases = [
        "hello world",
        "həˈloʊ wɝld",
        "bˈajit hˈu",
        "ʃeχajˈim",
    ]

    all_passed = True
    for text in test_cases:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        ids = encoded.ids

        # Check 1 token per character (excluding <unk>)
        expected = len(text)
        actual = len(tokens)

        status = "✓" if actual == expected else "⚠️"
        if actual != expected:
            all_passed = False

        print(f"\n  {status} '{text}'")
        print(f"     Tokens ({actual}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

        # Check for <unk> tokens
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"     ⚠️ {unk_count} unknown tokens")

    return all_passed


def test_special_tokens(tokenizer: Tokenizer):
    """Test that special tokens are handled correctly."""
    print("\n" + "=" * 60)
    print("2. SPECIAL TOKENS")
    print("=" * 60)

    # Test audio markers
    test = "hello<audio_start><snac_l1_100><snac_l2_50><snac_l3_25><audio_end>"
    encoded = tokenizer.encode(test)

    print(f"\n  Input: '{test}'")
    print(f"  Tokens: {encoded.tokens}")

    # Verify special tokens are whole
    expected_special = ["<audio_start>", "<snac_l1_100>", "<snac_l2_50>", "<snac_l3_25>", "<audio_end>"]
    found_special = [t for t in encoded.tokens if t.startswith("<") and t.endswith(">")]

    if found_special == expected_special:
        print(f"  ✓ Special tokens intact")
        return True
    else:
        print(f"  ⚠️ Expected: {expected_special}")
        print(f"     Got: {found_special}")
        return False


def test_snac_range(tokenizer: Tokenizer):
    """Test SNAC tokens across the full range."""
    print("\n" + "=" * 60)
    print("3. SNAC TOKEN RANGE")
    print("=" * 60)

    # Test edge cases
    test_tokens = [
        "<snac_l1_0>",
        "<snac_l1_4095>",
        "<snac_l2_0>",
        "<snac_l2_4095>",
        "<snac_l3_0>",
        "<snac_l3_4095>",
    ]

    all_valid = True
    for token in test_tokens:
        encoded = tokenizer.encode(token)
        if len(encoded.tokens) == 1 and encoded.tokens[0] == token:
            print(f"  ✓ {token} → ID {encoded.ids[0]}")
        else:
            print(f"  ⚠️ {token} → {encoded.tokens}")
            all_valid = False

    return all_valid


def test_roundtrip(tokenizer: Tokenizer):
    """Test encode → decode roundtrip."""
    print("\n" + "=" * 60)
    print("4. ENCODE/DECODE ROUNDTRIP")
    print("=" * 60)

    test = "həˈloʊ<audio_start><snac_l1_100><snac_l2_50><audio_end>"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)

    print(f"\n  Original: '{test}'")
    print(f"  Decoded:  '{decoded}'")

    # Note: decode may add spaces between tokens
    if test.replace(" ", "") == decoded.replace(" ", ""):
        print(f"  ✓ Roundtrip successful (ignoring spaces)")
        return True
    else:
        print(f"  ⚠️ Roundtrip mismatch")
        return False


def test_with_dataset(tokenizer: Tokenizer, dataset_path: str):
    """Test tokenizer with real dataset samples."""
    print("\n" + "=" * 60)
    print("5. DATASET SAMPLES")
    print("=" * 60)

    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f if line.strip()][:10]

    total_unk = 0
    for i, sample in enumerate(samples):
        text = sample["text"]
        encoded = tokenizer.encode(text)

        unk_count = encoded.tokens.count("<unk>")
        total_unk += unk_count

        status = "✓" if unk_count == 0 else "⚠️"
        print(f"\n  {status} Sample {i+1}: {text[:40]}...")
        print(f"     {len(encoded.tokens)} tokens, {unk_count} <unk>")

        if unk_count > 0:
            # Find which characters are unknown
            unk_chars = []
            for j, tok in enumerate(encoded.tokens):
                if tok == "<unk>":
                    unk_chars.append(text[j] if j < len(text) else "?")
            print(f"     Unknown chars: {unk_chars}")

    if total_unk == 0:
        print(f"\n  ✓ All samples tokenized without <unk>")
        return True
    else:
        print(f"\n  ⚠️ {total_unk} total <unk> tokens across samples")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="Path to tokenizer.json")
    parser.add_argument("dataset", nargs="?", help="Optional dataset.jsonl to test")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Vocab size: {tokenizer.get_vocab_size()}\n")

    results = []
    results.append(("Character-level", test_character_level(tokenizer)))
    results.append(("Special tokens", test_special_tokens(tokenizer)))
    results.append(("SNAC range", test_snac_range(tokenizer)))
    results.append(("Roundtrip", test_roundtrip(tokenizer)))

    if args.dataset:
        results.append(("Dataset", test_with_dataset(tokenizer, args.dataset)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "⚠️"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
