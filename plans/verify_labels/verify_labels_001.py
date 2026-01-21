# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers==4.51.1", "torch"]
# ///
"""
Verify label masking in TTSDataset.

The fix ensures that only audio token predictions contribute to training loss,
not text token predictions.

Usage: uv run plans/verify_labels/verify_labels_001.py dataset/dataset.jsonl
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import setup_tokenizer, TTSDataset


def test_label_masking(dataset_path, model_name, n_samples=3):
    """Verify labels are correctly masked."""
    print("Testing label masking in TTSDataset...\n")

    tokenizer = setup_tokenizer(model_name)
    dataset = TTSDataset(dataset_path, tokenizer, max_length=1024)

    audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")
    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

    print(f"<audio_start> ID: {audio_start_id}")
    print(f"<audio_end> ID: {audio_end_id}")
    print(f"Label ignore index: {TTSDataset.LABEL_IGNORE_INDEX}\n")

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        assert len(input_ids) == len(labels), f"Sample {i}: length mismatch"

        # Find <audio_start> position
        try:
            audio_start_pos = input_ids.index(audio_start_id)
        except ValueError:
            print(f"Sample {i}: SKIP (no <audio_start> - likely truncated)")
            continue

        # Verify all labels before and including <audio_start> are masked
        for j in range(audio_start_pos + 1):
            assert labels[j] == TTSDataset.LABEL_IGNORE_INDEX, \
                f"Sample {i}: Position {j} should be masked (got {labels[j]})"

        # Verify labels after <audio_start> match input_ids
        for j in range(audio_start_pos + 1, len(labels)):
            assert labels[j] == input_ids[j], \
                f"Sample {i}: Position {j} label mismatch ({labels[j]} != {input_ids[j]})"

        # Count masked vs unmasked
        n_masked = labels.count(TTSDataset.LABEL_IGNORE_INDEX)
        n_active = len(labels) - n_masked

        # Decode for visual inspection
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids[:audio_start_pos])
        audio_tokens = tokenizer.convert_ids_to_tokens(input_ids[audio_start_pos + 1:audio_start_pos + 8])

        print(f"Sample {i}:")
        print(f"  Total tokens: {len(input_ids)}")
        print(f"  Masked (text + <audio_start>): {n_masked}")
        print(f"  Active (audio + <audio_end>): {n_active}")
        print(f"  <audio_start> at position: {audio_start_pos}")
        print(f"  First text tokens: {text_tokens[:5]}...")
        print(f"  First audio tokens: {audio_tokens}...")
        print()

    print("All label masking tests passed!")


def test_edge_cases(model_name):
    """Test edge cases in label creation."""
    print("\nTesting edge cases...\n")

    tokenizer = setup_tokenizer(model_name)
    audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")

    # Create minimal dataset instance just to test _create_labels
    class MockDataset(TTSDataset):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.max_length = 1024
            self.audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")
            self.samples = []

    mock = MockDataset(tokenizer)

    # Test 1: Normal case
    input_ids = [1, 2, 3, audio_start_id, 100, 101, 102]
    labels = mock._create_labels(input_ids)
    assert labels == [-100, -100, -100, -100, 100, 101, 102], f"Normal case failed: {labels}"
    print("  1. Normal case: PASS")

    # Test 2: <audio_start> at beginning
    input_ids = [audio_start_id, 100, 101]
    labels = mock._create_labels(input_ids)
    assert labels == [-100, 100, 101], f"Start at beginning failed: {labels}"
    print("  2. <audio_start> at position 0: PASS")

    # Test 3: Only <audio_start> and <audio_end>
    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
    input_ids = [1, 2, audio_start_id, audio_end_id]
    labels = mock._create_labels(input_ids)
    assert labels == [-100, -100, -100, audio_end_id], f"Minimal audio failed: {labels}"
    print("  3. Minimal audio (just <audio_end>): PASS")

    # Test 4: No <audio_start> (truncated)
    input_ids = [1, 2, 3, 4, 5]
    labels = mock._create_labels(input_ids)
    assert all(l == -100 for l in labels), f"Missing <audio_start> should mask all: {labels}"
    print("  4. Missing <audio_start> (masks all): PASS")

    print("\nAll edge case tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Verify label masking")
    parser.add_argument("dataset", help="Path to dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("-n", type=int, default=3, help="Number of samples")
    args = parser.parse_args()

    try:
        test_label_masking(args.dataset, args.model, args.n)
        test_edge_cases(args.model)
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
