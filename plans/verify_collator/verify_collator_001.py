# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers==4.51.1", "torch"]
# ///
"""
Verify TTSDataCollator handles batching and padding correctly.

Usage: uv run plans/verify_collator/verify_collator_001.py dataset/dataset.jsonl
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import setup_tokenizer, TTSDataset, TTSDataCollator


def test_collator_padding(dataset_path, model_name, batch_size=4):
    """Test that collator properly pads batches."""
    print(f"Testing TTSDataCollator with batch_size={batch_size}...\n")

    tokenizer = setup_tokenizer(model_name)
    dataset = TTSDataset(dataset_path, tokenizer, max_length=1024)
    collator = TTSDataCollator(tokenizer=tokenizer)

    # Get samples of different lengths
    samples = [dataset[i] for i in range(batch_size)]
    lengths = [len(s["input_ids"]) for s in samples]

    print(f"Individual sample lengths: {lengths}")
    print(f"Max length in batch: {max(lengths)}")

    # Collate
    batch = collator(samples)

    print(f"\nBatch shapes:")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels:         {batch['labels'].shape}")

    # Verify shapes match
    assert batch["input_ids"].shape == batch["attention_mask"].shape, "input_ids/attention_mask shape mismatch"
    assert batch["input_ids"].shape == batch["labels"].shape, "input_ids/labels shape mismatch"
    assert batch["input_ids"].shape[0] == batch_size, f"Expected batch size {batch_size}"

    # Verify padding is correct
    padded_length = batch["input_ids"].shape[1]
    assert padded_length == max(lengths), f"Padded to {padded_length}, expected {max(lengths)}"
    print(f"\n  Padded to max length: {padded_length}")

    # Verify attention_mask is 0 where padded
    pad_token_id = tokenizer.pad_token_id
    for i, orig_len in enumerate(lengths):
        # Check padding in input_ids
        if orig_len < padded_length:
            pad_region = batch["input_ids"][i, orig_len:]
            assert (pad_region == pad_token_id).all(), f"Sample {i}: input_ids not padded correctly"

            # Check attention_mask is 0 for padding
            attn_pad = batch["attention_mask"][i, orig_len:]
            assert (attn_pad == 0).all(), f"Sample {i}: attention_mask not 0 for padding"

            # Check labels are -100 for padding
            label_pad = batch["labels"][i, orig_len:]
            assert (label_pad == -100).all(), f"Sample {i}: labels not -100 for padding"

    print("  Padding verified: input_ids, attention_mask, labels all correct")

    # Verify label masking is preserved (not just all -100)
    for i in range(batch_size):
        labels = batch["labels"][i].tolist()
        n_active = sum(1 for l in labels if l != -100)
        n_masked = sum(1 for l in labels if l == -100)
        print(f"  Sample {i}: {n_active} active labels, {n_masked} masked (including padding)")
        assert n_active > 0, f"Sample {i}: No active labels - masking broken"

    print("\nAll collator tests passed!")


def test_collator_single_sample(dataset_path, model_name):
    """Test collator works with single sample (no padding needed)."""
    print("\nTesting single sample batch...")

    tokenizer = setup_tokenizer(model_name)
    dataset = TTSDataset(dataset_path, tokenizer, max_length=1024)
    collator = TTSDataCollator(tokenizer=tokenizer)

    samples = [dataset[0]]
    batch = collator(samples)

    assert batch["input_ids"].shape[0] == 1, "Batch size should be 1"
    assert batch["input_ids"].shape[1] == len(dataset[0]["input_ids"]), "Should not pad single sample"

    print("  Single sample batch: PASS")


def test_collator_preserves_content(dataset_path, model_name):
    """Test that collator doesn't corrupt data."""
    print("\nTesting content preservation...")

    tokenizer = setup_tokenizer(model_name)
    dataset = TTSDataset(dataset_path, tokenizer, max_length=1024)
    collator = TTSDataCollator(tokenizer=tokenizer)

    # Get original samples
    originals = [
        {"input_ids": dataset[i]["input_ids"].copy(), "labels": dataset[i]["labels"].copy()}
        for i in range(3)
    ]

    # Need fresh samples since collator.pop() modifies them
    samples = [dataset[i] for i in range(3)]
    batch = collator(samples)

    for i, orig in enumerate(originals):
        orig_len = len(orig["input_ids"])
        # Check input_ids preserved (before padding)
        batch_ids = batch["input_ids"][i, :orig_len].tolist()
        assert batch_ids == orig["input_ids"], f"Sample {i}: input_ids corrupted"

        # Check labels preserved (before padding)
        batch_labels = batch["labels"][i, :orig_len].tolist()
        assert batch_labels == orig["labels"], f"Sample {i}: labels corrupted"

    print("  Content preservation: PASS")


def main():
    parser = argparse.ArgumentParser(description="Verify data collator")
    parser.add_argument("dataset", help="Path to dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size to test")
    args = parser.parse_args()

    try:
        test_collator_padding(args.dataset, args.model, args.batch_size)
        test_collator_single_sample(args.dataset, args.model)
        test_collator_preserves_content(args.dataset, args.model)

        print("\n" + "=" * 50)
        print("ALL COLLATOR TESTS PASSED")
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
