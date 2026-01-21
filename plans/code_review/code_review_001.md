# Sheen TTS Code Review Report

**Reviewed:** 2026-01-21
**Reviewer:** Claude
**Files Reviewed:** `src/config.py`, `src/data.py`, `src/train.py`, `src/infer.py`, `scripts/prepare_dataset.py`

---

## Executive Summary

The codebase implements a Qwen LLM + SNAC neural audio codec TTS system. The architecture is fundamentally sound, but there are several bugs and design issues that will cause problems during training and inference.

**Critical Issues:** 3
**Major Issues:** 6
**Minor Issues:** 8

---

## Critical Issues (Will Break Training/Inference)

### 1. ~~Loss Computed on Text Tokens - Training Won't Converge Properly~~ FIXED

**File:** `src/data.py:86`

```python
encoded["labels"] = encoded["input_ids"].copy()
```

**Problem:** The model is trained to predict ALL tokens, including the input text. This wastes model capacity on predicting text tokens, which isn't the goal. The model should only learn to predict audio tokens given text.

**Fix:** Mask text tokens by setting them to `-100` (HuggingFace's ignore index):

```python
# Find position of <audio_start> token
audio_start_id = self.tokenizer.convert_tokens_to_ids("<audio_start>")
audio_start_pos = encoded["input_ids"].index(audio_start_id)

labels = [-100] * (audio_start_pos + 1)  # Ignore text + <audio_start>
labels += encoded["input_ids"][audio_start_pos + 1:]  # Only predict audio tokens
encoded["labels"] = labels
```

---

### 2. ~~No DataCollator - Batched Training Will Crash~~ FIXED

**File:** `src/train.py:53-59`

**Problem:** No `DataCollatorForLanguageModeling` or custom collator is provided. The `Trainer` needs to pad sequences in a batch to the same length. Without a collator that handles padding, training with `batch_size > 1` will crash.

**Additionally:** `tokenizer.pad_token` is never set. Qwen's tokenizer may not have one by default.

**Fix:**

```python
from transformers import DataCollatorForLanguageModeling

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Use collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    ...
    data_collator=data_collator,
)
```

---

### 3. ~~argparse `store_true` with `default=True` Bug~~ FIXED

**File:** `src/config.py:31-32`

```python
parser.add_argument("--fp16", action="store_true", default=True)
parser.add_argument("--grad-checkpoint", action="store_true", default=True)
```

**Problem:** `action="store_true"` means the flag is `False` when absent and `True` when present. Setting `default=True` creates contradictory behavior - users cannot disable these features via CLI.

**Fix:** Use `BooleanOptionalAction` (Python 3.9+) or separate flags:

```python
parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
```

This allows `--fp16` to enable and `--no-fp16` to disable.

---

## Major Issues (Significant Problems)

### 4. Empty Generation Crashes Inference

**File:** `src/infer.py:52-53`

```python
codes_tensor = [torch.tensor(c).unsqueeze(0).to(device) for c in codes]
audio = snac.decode(codes_tensor)
```

**Problem:** If the model fails to generate any SNAC tokens (generates `<audio_end>` immediately, or only garbage), `codes` will be `[[], [], []]`. SNAC's decode will crash on empty tensors.

**Fix:** Add validation:

```python
if not codes[0] or not codes[1] or not codes[2]:
    print("Error: No audio tokens generated")
    return None

# Also validate ratio
if len(codes[1]) < len(codes[0]) * 2 or len(codes[2]) < len(codes[0]) * 4:
    print("Warning: Incomplete SNAC frames generated")
```

---

### 5. ~~No Shuffle in Train/Eval Split~~ FIXED

**File:** `src/train.py:29-31`

```python
split_idx = int(len(dataset) * 0.9)
train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
```

**Problem:** The split is sequential - first 90% for training, last 10% for eval. If the dataset has any ordering (by speaker, topic, recording date, or length), this creates a biased split. The eval set won't represent the training distribution.

**Fix:** Shuffle indices:

```python
import random
indices = list(range(len(dataset)))
random.seed(42)  # Reproducible
random.shuffle(indices)
train_dataset = torch.utils.data.Subset(dataset, indices[:split_idx])
eval_dataset = torch.utils.data.Subset(dataset, indices[split_idx:])
```

---

### 6. Relative Imports Will Fail

**Files:** `src/data.py:6`, `src/train.py:5-6`, `src/infer.py:9`

```python
from config import SPECIAL_TOKENS, SNAC_TOKENS  # data.py
from config import get_args  # train.py
from data import deinterleave_snac_tokens  # infer.py
```

**Problem:** These are bare imports that assume `src/` is in `PYTHONPATH`. Running from project root (`uv run src/train.py`) will fail with `ModuleNotFoundError`.

**Fix:** Either:
1. Add `__init__.py` and use relative imports: `from .config import ...`
2. Or run with `python -m src.train`
3. Or make `src` a proper package in `pyproject.toml`

---

### 7. Dataset Loads Entire File Into Memory

**File:** `src/data.py:68-69`

```python
with open(path) as f:
    self.samples = [json.loads(line) for line in f if line.strip()]
```

**Problem:** For large datasets (100k+ samples), this loads everything into memory. Each sample contains SNAC codes which can be 500+ integers. With 100k samples, this could be several GB of RAM.

**Fix:** Use memory-mapped or lazy loading:

```python
def __init__(self, path, tokenizer, max_length=1024):
    self.path = path
    self.tokenizer = tokenizer
    self.max_length = max_length

    # Only store file offsets
    self.offsets = []
    with open(path, 'rb') as f:
        while line := f.readline():
            self.offsets.append(f.tell() - len(line))

def __getitem__(self, idx):
    with open(self.path, 'r') as f:
        f.seek(self.offsets[idx])
        sample = json.loads(f.readline())
    ...
```

---

### 8. ~~Deprecated TrainingArguments Parameter~~ FIXED

**File:** `src/train.py:47`

```python
evaluation_strategy="steps",
```

**Problem:** `evaluation_strategy` is deprecated in transformers >= 4.46. Should use `eval_strategy`.

**Fix:**

```python
eval_strategy="steps",
```

---

### 9. No SNAC Code Range Validation

**File:** `src/data.py:38-46`

```python
tokens.extend([
    f"<snac_l1_{l1[i]}>",
    ...
])
```

**Problem:** No validation that codes are within 0-4095 range. If the dataset contains invalid codes, this creates tokens like `<snac_l1_5000>` which don't exist in the vocabulary and will cause tokenization errors.

**Fix:**

```python
def interleave_snac_codes(snac_codes):
    l1, l2, l3 = snac_codes
    tokens = []

    for i in range(len(l1)):
        # Validate ranges
        if not (0 <= l1[i] < 4096):
            raise ValueError(f"Invalid L1 code: {l1[i]}")
        # ... similar for l2, l3
```

---

## Minor Issues

### 10. Model Loaded Every Inference Call

**File:** `src/infer.py:17-19`

The model, tokenizer, and SNAC decoder are loaded fresh for each `generate_speech()` call. For multiple generations, this is extremely slow.

**Recommendation:** Create a class or load models once externally.

---

### 11. Hardcoded SNAC Model Path

**Files:** `src/infer.py:19`, `scripts/prepare_dataset.py:88`

```python
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
```

Should be configurable via argument for offline/custom models.

---

### 12. No Training Resume Capability

**File:** `src/train.py`

No `resume_from_checkpoint` support. If training crashes after hours, must restart from scratch.

**Fix:** Add to TrainingArguments:

```python
resume_from_checkpoint=True,  # or specific path
```

---

### 13. scipy.signal.resample Is Slow

**File:** `scripts/prepare_dataset.py:32`

`scipy.signal.resample` uses FFT-based resampling which is slow for audio. For 3000+ files, this significantly increases preprocessing time.

**Recommendation:** Use `scipy.signal.resample_poly` or `librosa.resample` for 10-50x speedup.

---

### 14. No Progress Saving in prepare_dataset.py

If dataset preparation crashes at 80%, must restart from beginning.

**Recommendation:** Track processed IDs and skip already-processed entries.

---

### 15. max_new_tokens May Be Insufficient

**File:** `src/infer.py:30`

```python
max_new_tokens=1000,
```

With 7 tokens per frame and ~84 tokens/second of audio, this limits output to ~12 seconds. For longer utterances, audio will be cut off.

**Recommendation:** Make configurable or calculate based on expected duration.

---

### 16. No Input Validation in Inference

**File:** `src/infer.py:12`

No validation that `text` is non-empty or `model_path` exists.

---

### 17. Emoji in Output

**File:** `scripts/prepare_dataset.py:111`

```python
print(f"✓ Done! Created {output_file}")
```

Minor inconsistency with project style (CLAUDE.md says avoid emojis).

---

## Architecture Observations

### What's Good

1. **Correct approach:** LLM-based TTS with frame-interleaved SNAC codes is a valid architecture
2. **Clean separation:** config/data/train/infer are properly modularized
3. **Comprehensive verification:** The verify_*.py scripts are excellent for testing components
4. **Gradient checkpointing:** Good for memory efficiency on smaller GPUs

### Potential Improvements (Not Bugs)

1. **Consider using `bf16` instead of `fp16`** - Better numerical stability for training
2. **Add audio quality metrics** - Current eval only uses perplexity, not MOS/PESQ
3. **Consider learning rate scheduler** - Cosine annealing often helps
4. **Add seed for reproducibility** - No `set_seed()` call

---

## Recommended Fix Priority

1. **Fix label masking** (Critical - wastes training compute)
2. **Add DataCollator** (Critical - batch training broken)
3. **Fix argparse flags** (Critical - can't configure training)
4. **Shuffle train/eval split** (Major - biased evaluation)
5. **Fix imports** (Major - code won't run as-is)
6. **Add empty generation check** (Major - inference crashes)
7. **Validate SNAC ranges** (Major - silent data corruption)
8. **Fix deprecated parameter** (Minor - warning spam)

---

## Summary

The codebase has a sound architecture but contains several bugs that will prevent successful training. The most critical issue is that the model trains on text tokens, not just audio tokens - this will significantly hurt convergence. Additionally, the missing DataCollator means batched training is broken.

Before starting training:
1. Fix the 3 critical issues
2. Fix the train/eval shuffle to get valid metrics
3. Fix the imports so the code actually runs

Estimated time to fix all issues: Medium effort (mostly straightforward fixes).
