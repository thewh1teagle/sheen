# Qwen + SNAC TTS Model Plan (v2)

## Overview

Simplified implementation using Qwen's tokenizer with added SNAC tokens. Split into logical modules.

---

## File Structure

```
sheen/
├── scripts/
│   └── prepare_dataset.py   # Keep as-is (already good)
├── src/
│   ├── config.py            # Argparse + constants
│   ├── data.py              # Dataset + formatting
│   ├── train.py             # Training entry point
│   └── infer.py             # Inference entry point
```

---

## Module Specifications

### `src/config.py`

Argparse configuration and constants.

```python
"""Training configuration."""
import argparse

# SNAC constants
SNAC_VOCAB_SIZE = 4096
SNAC_LAYERS = 3
SNAC_TOKENS = [f"<snac_l{l}_{c}>" for l in [1, 2, 3] for c in range(SNAC_VOCAB_SIZE)]
SPECIAL_TOKENS = ["<audio_start>", "<audio_end>"]


def get_args():
    parser = argparse.ArgumentParser(description="Train Qwen+SNAC TTS")

    # Required
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base model")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)

    # Hardware
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--grad-checkpoint", action="store_true", default=True)

    # Logging
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)

    return parser.parse_args()
```

---

### `src/data.py`

Dataset loading and SNAC token formatting.

```python
"""Dataset and SNAC token formatting."""
import json
from torch.utils.data import Dataset

from config import SPECIAL_TOKENS, SNAC_TOKENS


def setup_tokenizer(model_name):
    """Load Qwen tokenizer and add SNAC tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.add_tokens(SNAC_TOKENS)
    return tokenizer


def interleave_snac_codes(snac_codes):
    """
    Convert SNAC codes to frame-interleaved token strings.

    Each frame: 1 L1 + 2 L2 + 4 L3 = 7 tokens

    Args:
        snac_codes: [layer1_codes, layer2_codes, layer3_codes]

    Returns:
        List of token strings like ["<snac_l1_123>", "<snac_l2_456>", ...]
    """
    l1, l2, l3 = snac_codes
    tokens = []

    for i in range(len(l1)):
        # Check bounds for L2 and L3
        if i * 2 + 1 >= len(l2) or i * 4 + 3 >= len(l3):
            break

        # Frame: 1 + 2 + 4 = 7 tokens
        tokens.extend([
            f"<snac_l1_{l1[i]}>",
            f"<snac_l2_{l2[i*2]}>",
            f"<snac_l2_{l2[i*2+1]}>",
            f"<snac_l3_{l3[i*4]}>",
            f"<snac_l3_{l3[i*4+1]}>",
            f"<snac_l3_{l3[i*4+2]}>",
            f"<snac_l3_{l3[i*4+3]}>",
        ])

    return tokens


def format_sample(text, snac_codes):
    """
    Format training sample.

    Format: {text} <audio_start> {audio_tokens} <audio_end>
    """
    audio_tokens = interleave_snac_codes(snac_codes)
    return f"{text} <audio_start> {' '.join(audio_tokens)} <audio_end>"


class TTSDataset(Dataset):
    """Dataset for TTS training."""

    def __init__(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = format_sample(sample["text"], sample["snac_codes"])

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded


def deinterleave_snac_tokens(tokens):
    """
    Convert token strings back to SNAC codes.

    Args:
        tokens: List of token strings like ["<snac_l1_123>", ...]

    Returns:
        [layer1_codes, layer2_codes, layer3_codes]
    """
    l1, l2, l3 = [], [], []

    for token in tokens:
        if not token.startswith("<snac_l"):
            continue

        # Parse "<snac_l1_123>" -> layer=1, code=123
        parts = token[1:-1].split("_")  # Remove < > and split
        layer = int(parts[1][1])  # "l1" -> 1
        code = int(parts[2])

        if layer == 1:
            l1.append(code)
        elif layer == 2:
            l2.append(code)
        else:
            l3.append(code)

    return [l1, l2, l3]
```

---

### `src/train.py`

Training entry point.

```python
"""Training script for Qwen+SNAC TTS."""
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from config import get_args
from data import setup_tokenizer, TTSDataset


def main():
    args = get_args()

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    # Tokenizer
    tokenizer = setup_tokenizer(args.model)
    print(f"Vocab size: {len(tokenizer)}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Dataset
    dataset = TTSDataset(args.dataset, tokenizer, args.max_length)
    split_idx = int(len(dataset) * 0.9)
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"Saved to {args.output}/final")


if __name__ == "__main__":
    main()
```

---

### `src/infer.py`

Inference script.

```python
"""Inference for Qwen+SNAC TTS."""
import argparse
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

from data import deinterleave_snac_tokens


def generate_speech(text, model_path, output_path="output.wav"):
    """Generate speech from text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    # Generate
    prompt = f"{text} <audio_start>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            eos_token_id=audio_end_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    # Extract audio tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]  # Skip prompt
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    # Filter to just SNAC tokens (before <audio_end>)
    snac_tokens = []
    for t in tokens:
        if t == "<audio_end>":
            break
        if t.startswith("<snac_l"):
            snac_tokens.append(t)

    # Convert back to codes
    codes = deinterleave_snac_tokens(snac_tokens)

    # Decode audio
    codes_tensor = [torch.tensor(c).unsqueeze(0).to(device) for c in codes]
    audio = snac.decode(codes_tensor)
    audio_np = audio.squeeze().cpu().numpy()

    sf.write(output_path, audio_np, 24000)
    print(f"Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Input text/phonemes")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    args = parser.parse_args()

    generate_speech(args.text, args.model, args.output)


if __name__ == "__main__":
    main()
```

---

## Sequence Format

```
{phonemes} <audio_start> <snac_l1_X> <snac_l2_Y> <snac_l2_Z> <snac_l3_A> <snac_l3_B> <snac_l3_C> <snac_l3_D> ... <audio_end>
           └─ delimiter ─┘ └──────────────────── 7 tokens per frame ────────────────────────────┘              └─ stop token ─┘
```

---

## Files to Delete

```
DELETE:
- src/tokenizer.py          # Replaced by data.py + Qwen tokenizer
- src/trainer.py            # Merged into train.py
- scripts/prepare_tokenizer.py  # Not needed (use Qwen tokenizer)

KEEP:
- scripts/prepare_dataset.py  # Already good

REWRITE:
- src/config.py   # Simplify argparse
- src/train.py    # Use Qwen tokenizer approach

CREATE:
- src/data.py     # Dataset + SNAC formatting
- src/infer.py    # Inference
```

---

## Usage

```bash
# 1. Prepare dataset (already done)
uv run scripts/prepare_dataset.py --dataset-dir dataset/

# 2. Train
python src/train.py \
    --dataset dataset/dataset.jsonl \
    --output outputs/tts \
    --epochs 5

# 3. Inference
python src/infer.py \
    --model outputs/tts/final \
    --text "həˈloʊ wɝld" \
    --output hello.wav
```

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Files | 5 | 4 |
| Lines | ~700 | ~250 |
| Custom tokenizer | Yes (230 lines) | No (use Qwen's) |
| Qwen text understanding | Lost | Preserved |
