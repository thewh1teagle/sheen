# Qwen + SNAC TTS Model Plan

## Research Review

The junior agent's research is **mostly correct** with a few issues to address:

### What's Correct

1. **LLM + Codec Architecture** - Using Qwen as the text-to-token predictor and SNAC as the audio codec is a valid, modern approach inspired by VALL-E and similar models.

2. **SNAC Structure** - SNAC 24kHz produces 3 layers at ~12/24/48 Hz rates (1:2:4 ratio). Each layer has 4096 codes (0-4095). Total: 12,288 unique audio codes.

3. **Frame Interleaving** - Grouping tokens as 1 L1 + 2 L2 + 4 L3 = 7 tokens per frame is correct and matches the temporal ratios.

4. **Special Tokens** - Using `<audio_start>` and `<audio_end>` (or `<eos>`) as delimiters is standard practice.

5. **Causal LM Training** - Training the model to predict the next token (text → audio tokens) is the correct approach.

### Issues Found

1. **Tokenizer Approach Mismatch** - Research says "extend Qwen's tokenizer with `add_tokens`" but current code builds a **completely custom tokenizer from scratch**. This loses Qwen's pretrained text understanding.

2. **Phoneme Source** - Code downloads piper phonemes but dataset uses IPA characters. The vocab may not match the actual phoneme format in the data.

3. **No BOS Token Used** - The sequence format doesn't include `<bos>`, which Qwen expects for generation.

---

## Current Code Problems (Over-Engineering)

| File | Issue |
|------|-------|
| `src/config.py` | Duplicates everything - dataclass mirrors argparse exactly. 200 lines for what could be 20. |
| `src/tokenizer.py` | Custom tokenizer from scratch. Loses Qwen's text capabilities. 230 lines of unnecessary abstraction. |
| `src/trainer.py` | Custom collator when HF provides one. `compute_metrics` computes accuracy which is meaningless for TTS. |
| `scripts/prepare_tokenizer.py` | Downloads piper phonemes but dataset uses different IPA format. |

**Total: ~700 lines that could be ~200 lines.**

---

## Simplified Architecture

### Option A: Use Qwen's Tokenizer (Recommended)

```
Qwen tokenizer handles text naturally
+ Add 12,288 SNAC tokens (<snac_l1_0> ... <snac_l3_4095>)
+ Add <audio_start>, <audio_end>
= Done
```

**Benefits:**
- Qwen's pretrained text understanding preserved
- Works with any text (phonemes, orthographic, mixed)
- Standard HuggingFace workflow

### Option B: Custom Vocab (Current approach, simplified)

Keep character-level vocab but drastically simplify the code.

**Recommendation: Option A** - It's simpler and leverages the pretrained model better.

---

## Simplified Implementation Plan

### Step 1: Dataset Preparation (`scripts/prepare_dataset.py`)

**Current code is mostly fine.** Minor fix needed:
- Already saves `{"text": phonemes, "snac_codes": [[l1], [l2], [l3]]}`
- This is correct

**No changes needed.**

### Step 2: Training Script (Single File)

Replace `src/config.py`, `src/tokenizer.py`, `src/trainer.py`, `src/train.py` with **one file**:

```python
# train.py (~150 lines)
"""
Usage: python train.py --dataset dataset/dataset.jsonl --output outputs/tts
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# --- Constants ---
SNAC_TOKENS = [f"<snac_l{l}_{c}>" for l in [1,2,3] for c in range(4096)]
SPECIAL_TOKENS = ["<audio_start>", "<audio_end>"]

# --- Tokenizer Setup ---
def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.add_tokens(SNAC_TOKENS)
    return tokenizer

# --- Data Formatting ---
def interleave_codes(snac_codes):
    """Convert [l1, l2, l3] to frame-interleaved token strings."""
    l1, l2, l3 = snac_codes
    tokens = []
    for i in range(len(l1)):
        if i*2+1 >= len(l2) or i*4+3 >= len(l3):
            break
        tokens.extend([
            f"<snac_l1_{l1[i]}>",
            f"<snac_l2_{l2[i*2]}>", f"<snac_l2_{l2[i*2+1]}>",
            f"<snac_l3_{l3[i*4]}>", f"<snac_l3_{l3[i*4+1]}>",
            f"<snac_l3_{l3[i*4+2]}>", f"<snac_l3_{l3[i*4+3]}>",
        ])
    return tokens

def format_sample(text, snac_codes):
    """Format: text <audio_start> audio_tokens <audio_end>"""
    audio_tokens = interleave_codes(snac_codes)
    return text + " <audio_start> " + " ".join(audio_tokens) + " <audio_end>"

# --- Dataset ---
class TTSDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
        self.samples = [json.loads(l) for l in open(path) if l.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        text = format_sample(s["text"], s["snac_codes"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len)
        enc["labels"] = enc["input_ids"].copy()
        return enc

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="outputs/tts")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # Setup
    tokenizer = setup_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dataset = TTSDataset(args.dataset, tokenizer)
    split = int(len(dataset) * 0.9)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    # Train
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")

if __name__ == "__main__":
    main()
```

### Step 3: Inference Script

```python
# infer.py (~50 lines)
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import torch
import soundfile as sf

def generate_speech(text, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    # Generate tokens
    prompt = text + " <audio_start>"
    inputs = tokenizer(prompt, return_tensors="pt")

    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        eos_token_id=audio_end_id,
        do_sample=False,
    )

    # Parse output tokens back to SNAC codes
    tokens = tokenizer.convert_ids_to_tokens(outputs[0])
    # ... deinterleave to [l1, l2, l3]
    # audio = snac.decode(codes)
    # sf.write("output.wav", audio, 24000)
```

---

## Sequence Format

```
<text_tokens> <audio_start> <snac_l1_X> <snac_l2_Y> <snac_l2_Z> <snac_l3_A> <snac_l3_B> <snac_l3_C> <snac_l3_D> ... <audio_end>
```

- **Text tokens**: Qwen's tokenizer handles this (subword tokenization)
- **<audio_start>**: Marks transition from text to audio
- **Audio frames**: 7 tokens per frame (1+2+4)
- **<audio_end>**: Stop token for generation

---

## Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen2.5-0.5B | Small enough to train on consumer GPU, powerful enough for TTS |
| Max seq length | 1024 | LJSpeech avg ~5-7s → ~500-700 audio tokens + text |
| Learning rate | 2e-5 | Standard for LLM fine-tuning |
| Batch size | 4 (+ grad accum) | Memory constraints |
| Epochs | 3-10 | Monitor eval loss |
| Loss | Full sequence | Including text is fine, model adapts |

---

## Files to Delete

After implementing the simplified version:

```
DELETE:
- src/config.py
- src/tokenizer.py
- src/trainer.py
- scripts/prepare_tokenizer.py

KEEP:
- scripts/prepare_dataset.py (already good)

CREATE:
- train.py (simplified training)
- infer.py (inference)
```

---

## Summary

The junior research is accurate on the core concepts. The main issues are:

1. **Over-engineering**: 700+ lines → ~200 lines possible
2. **Tokenizer mismatch**: Custom vocab loses Qwen's text understanding
3. **Unnecessary abstractions**: Config dataclass, custom collator, metrics

**Recommendation**: Rewrite with Option A (use Qwen's tokenizer + add SNAC tokens). This is simpler, more correct, and leverages the pretrained model better.
