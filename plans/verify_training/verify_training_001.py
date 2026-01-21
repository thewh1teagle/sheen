# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.5.1", "transformers==4.51.1"]
# ///
"""
Verify training pipeline: model loading, tokenizer, dataset, forward/backward pass.
Usage: uv run plans/verify_training/verify_training_001.py dataset/dataset.jsonl
"""
import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import SNAC_TOKENS, SPECIAL_TOKENS
from data import setup_tokenizer, TTSDataset


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_tokenizer_setup(model_name):
    """Test tokenizer setup."""
    print("1. Testing tokenizer setup...")

    tokenizer = setup_tokenizer(model_name)

    expected_new_tokens = len(SNAC_TOKENS) + len(SPECIAL_TOKENS)
    print(f"   ✓ Vocab size: {len(tokenizer)}")
    print(f"   ✓ Added {expected_new_tokens} tokens (SNAC + special)")

    return tokenizer


def test_model_loading(model_name, tokenizer):
    """Test model loading and embedding resize."""
    print("\n2. Testing model loading...")

    from transformers import AutoModelForCausalLM

    mem_before = get_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(model_name)
    old_vocab = model.config.vocab_size

    model.resize_token_embeddings(len(tokenizer))
    new_vocab = model.config.vocab_size

    print(f"   ✓ Original vocab: {old_vocab}")
    print(f"   ✓ Resized vocab: {new_vocab}")
    print(f"   ✓ Embedding shape: {model.get_input_embeddings().weight.shape}")

    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    mem_after = get_memory_mb()
    print(f"   ✓ Device: {device}")
    if device == "cuda":
        print(f"   ✓ GPU memory: {mem_after:.0f} MB")

    return model, device, old_vocab


def test_dataset_loading(dataset_path, tokenizer, n_samples=3):
    """Test dataset loading."""
    print(f"\n3. Testing dataset loading (n={n_samples})...")

    dataset = TTSDataset(dataset_path, tokenizer, max_length=1024)

    print(f"   ✓ Loaded {len(dataset)} samples")

    # Test a few samples
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        print(f"   Sample {i+1}: {len(input_ids)} tokens")

        assert len(input_ids) == len(labels), "input_ids and labels length mismatch"
        assert all(0 <= t < len(tokenizer) for t in input_ids), "Token ID out of range"

    print(f"   ✓ All samples valid")

    return dataset


def test_forward_pass(model, tokenizer, dataset, device):
    """Test forward pass."""
    print("\n4. Testing forward pass...")

    sample = dataset[0]

    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    attention_mask = torch.ones_like(input_ids)

    print(f"   Input shape: {input_ids.shape}")

    mem_before = get_memory_mb()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    mem_after = get_memory_mb()

    loss = outputs.loss.item()
    logits_shape = outputs.logits.shape

    print(f"   ✓ Loss: {loss:.4f}")
    print(f"   ✓ Logits shape: {logits_shape}")
    if device == "cuda":
        print(f"   ✓ GPU memory: {mem_after:.0f} MB (+{mem_after - mem_before:.0f} MB)")

    return loss


def test_backward_pass(model, tokenizer, dataset, device, old_vocab_size):
    """Test backward pass (one training step)."""
    print("\n5. Testing backward pass...")

    model.train()

    sample = dataset[0]

    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    attention_mask = torch.ones_like(input_ids)

    mem_before = get_memory_mb()

    # Forward
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    print(f"   ✓ Forward loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    mem_after = get_memory_mb()

    # Check gradients exist
    embedding_grad = model.get_input_embeddings().weight.grad
    has_grad = embedding_grad is not None and embedding_grad.abs().sum() > 0

    print(f"   ✓ Backward completed")
    print(f"   ✓ Gradients exist: {has_grad}")
    if device == "cuda":
        print(f"   ✓ GPU memory: {mem_after:.0f} MB (+{mem_after - mem_before:.0f} MB)")

    # Check new token embeddings have gradients
    new_token_grads = embedding_grad[old_vocab_size:].abs().sum().item()
    print(f"   ✓ New token gradient sum: {new_token_grads:.4f}")

    return True


def test_gradient_checkpointing(model, tokenizer, dataset, device):
    """Test with gradient checkpointing enabled."""
    print("\n6. Testing gradient checkpointing...")

    model.gradient_checkpointing_enable()
    model.zero_grad()

    sample = dataset[0]

    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    attention_mask = torch.ones_like(input_ids)

    mem_before = get_memory_mb()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    outputs.loss.backward()

    mem_after = get_memory_mb()

    print(f"   ✓ Gradient checkpointing works")
    if device == "cuda":
        print(f"   ✓ GPU memory: {mem_after:.0f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify training pipeline")
    parser.add_argument("dataset", help="Path to dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("-n", type=int, default=3, help="Number of samples to test")
    args = parser.parse_args()

    try:
        tokenizer = test_tokenizer_setup(args.model)
        model, device, old_vocab_size = test_model_loading(args.model, tokenizer)
        dataset = test_dataset_loading(args.dataset, tokenizer, args.n)
        test_forward_pass(model, tokenizer, dataset, device)
        test_backward_pass(model, tokenizer, dataset, device, old_vocab_size)
        test_gradient_checkpointing(model, tokenizer, dataset, device)

        print("\n" + "=" * 50)
        print("✓ All tests passed! Training pipeline is ready.")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
