# Qwen + SNAC TTS Architecture

## Overview

This is a single-speaker Text-to-Speech (TTS) system that combines:
- **Qwen 2.5** - A pretrained language model for sequence modeling
- **SNAC** - A neural audio codec for audio tokenization/reconstruction

The core idea: treat TTS as a language modeling task where the model learns to predict audio tokens given text input.

```
Text (phonemes) → [Qwen LLM] → Audio Tokens → [SNAC Decoder] → Waveform
```

---

## SNAC Audio Codec

### What is SNAC?

SNAC (Multi-Scale Neural Audio Codec) compresses audio into discrete tokens at multiple temporal resolutions. We use the 24kHz variant (`hubertsiuzdak/snac_24khz`).

### Hierarchical Structure

SNAC produces **3 layers** of tokens at different frame rates:

| Layer | Frame Rate | Tokens/Second | Codebook Size |
|-------|------------|---------------|---------------|
| L1    | ~12 Hz     | 12            | 4096 (0-4095) |
| L2    | ~24 Hz     | 24            | 4096 (0-4095) |
| L3    | ~48 Hz     | 48            | 4096 (0-4095) |

**Ratio: 1:2:4** - For every L1 token, there are 2 L2 tokens and 4 L3 tokens.

### Why Multiple Layers?

- **L1**: Captures coarse structure (prosody, phoneme boundaries)
- **L2**: Captures mid-level details (formants, transitions)
- **L3**: Captures fine details (texture, high frequencies)

### Example

For 1 second of audio at 24kHz (24,000 samples):
- L1: 12 tokens
- L2: 24 tokens
- L3: 48 tokens
- **Total: 84 tokens/second**

---

## Dataset Preparation

### Input Format

LJSpeech-style dataset:
```
dataset/
├── metadata.csv      # id|phonemes format
└── wav/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

### Processing Pipeline (`scripts/prepare_dataset.py`)

```
Audio File (.wav)
       ↓
Load + Resample to 24kHz + Mono
       ↓
SNAC Encode
       ↓
3 Lists: [L1_codes, L2_codes, L3_codes]
       ↓
Save as JSONL
```

### Output Format (`dataset.jsonl`)

```json
{
  "text": "həˈloʊ wɝld",
  "snac_codes": [
    [1234, 567, 890, ...],      // L1 codes (~12/sec)
    [111, 222, 333, 444, ...],  // L2 codes (~24/sec)
    [11, 22, 33, 44, 55, ...]   // L3 codes (~48/sec)
  ]
}
```

---

## Token Interleaving

### Why Interleave?

Instead of concatenating layers sequentially (all L1, then all L2, then all L3), we **interleave by frame**. This:
1. Maintains temporal alignment
2. Enables streaming generation
3. Groups related information together

### Frame Structure

Each frame = 7 tokens:
```
[L1] [L2] [L2] [L3] [L3] [L3] [L3]
 1  +  2      +      4        = 7 tokens per frame
```

### Token Format

Each SNAC code becomes a unique token string:
```
<snac_l1_1234>   # Layer 1, code 1234
<snac_l2_567>    # Layer 2, code 567
<snac_l3_89>     # Layer 3, code 89
```

Total audio vocabulary: 3 layers × 4096 codes = **12,288 tokens**

### Interleaving Example

Given codes:
```
L1: [100, 200]
L2: [10, 11, 20, 21]
L3: [1, 2, 3, 4, 5, 6, 7, 8]
```

Interleaved sequence:
```
Frame 1: <snac_l1_100> <snac_l2_10> <snac_l2_11> <snac_l3_1> <snac_l3_2> <snac_l3_3> <snac_l3_4>
Frame 2: <snac_l1_200> <snac_l2_20> <snac_l2_21> <snac_l3_5> <snac_l3_6> <snac_l3_7> <snac_l3_8>
```

---

## Training Sequence Format

### Full Sequence Structure

```
{phoneme_text} <audio_start> {interleaved_audio_tokens} <audio_end>
```

### Example

```
həˈloʊ wɝld <audio_start> <snac_l1_100> <snac_l2_10> <snac_l2_11> <snac_l3_1> ... <audio_end>
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<audio_start>` | Marks transition from text to audio |
| `<audio_end>` | End of audio / stop token for generation |

### Tokenization

1. **Text**: Tokenized by Qwen's tokenizer (subword/BPE)
2. **Special tokens**: Added to Qwen's vocabulary
3. **SNAC tokens**: Added to Qwen's vocabulary (12,288 new tokens)

Final vocab size: Qwen base (~152k) + 12,290 = ~164k tokens

---

## Model Architecture

### Base Model

Qwen 2.5-0.5B (or larger variants):
- Transformer decoder (causal LM)
- Pretrained on text
- Extended vocabulary for audio tokens

### Modifications

1. **Tokenizer extension**: Add SNAC + special tokens
2. **Embedding resize**: Expand embedding matrix for new tokens
3. **No architectural changes**: Standard causal LM training

### Why This Works

- Qwen already understands text/language
- Audio tokens are learned from scratch during fine-tuning
- The model learns the mapping: text patterns → audio token patterns

---

## Training

### Objective

Standard causal language modeling (next token prediction):

```
Loss = CrossEntropy(predicted_tokens, actual_tokens)
```

The model learns to predict:
1. `<audio_start>` after text
2. Each audio token given previous tokens
3. `<audio_end>` when audio should stop

### Data Flow

```
Input:  [text_tokens] [<audio_start>] [audio_tokens] [<audio_end>]
Labels: [text_tokens] [<audio_start>] [audio_tokens] [<audio_end>]
        ↑ shifted by 1 for next-token prediction
```

### Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Max sequence length | 1024 |
| Precision | FP16 |

---

## Inference

### Generation Pipeline

```
1. Input: "həˈloʊ wɝld"

2. Tokenize: [text_ids...]

3. Add prompt: [text_ids...] + [<audio_start>]

4. Generate autoregressively until <audio_end>:
   → <snac_l1_X> <snac_l2_Y> <snac_l2_Z> <snac_l3_A> ...

5. Extract audio tokens (between <audio_start> and <audio_end>)

6. Deinterleave to [L1, L2, L3] codes

7. SNAC decode → waveform

8. Save as .wav (24kHz)
```

### Deinterleaving

Reverse of interleaving - group every 7 tokens:
```
Tokens: [L1, L2, L2, L3, L3, L3, L3, L1, L2, L2, ...]
         └─────── frame 1 ───────┘  └─── frame 2 ──...

L1: [first of each frame]
L2: [2nd and 3rd of each frame]
L3: [4th-7th of each frame]
```

---

## Sequence Length Estimation

For a typical utterance:

| Component | Tokens |
|-----------|--------|
| Text (phonemes) | ~50-100 |
| `<audio_start>` | 1 |
| Audio (5 sec) | 5 × 12 × 7 = 420 |
| `<audio_end>` | 1 |
| **Total** | ~470-520 |

Max sequence length of 1024 handles ~10-12 seconds of audio.

---

## File Structure

```
sheen/
├── src/
│   ├── config.py      # Constants + argparse
│   ├── data.py        # Dataset + tokenization
│   ├── train.py       # Training entry point
│   └── infer.py       # Inference entry point
├── scripts/
│   └── prepare_dataset.py  # Audio → SNAC codes
└── plans/
    ├── architecture/       # This document
    ├── verify_dataset/     # Dataset validation
    ├── verify_tokenizer/   # Tokenizer tests
    ├── verify_snac/        # SNAC decode tests
    └── verify_training/    # Training pipeline tests
```

---

## Constants Reference

```python
SNAC_VOCAB_SIZE = 4096      # Codes per layer
SNAC_LAYERS = 3             # Number of layers
SNAC_SAMPLE_RATE = 24000    # Audio sample rate
TOKENS_PER_FRAME = 7        # 1 + 2 + 4
FRAMES_PER_SECOND = 12      # L1 rate = frame rate
AUDIO_TOKENS_PER_SEC = 84   # 12 + 24 + 48
```

---

## Reconstruction Quality

SNAC at 24kHz achieves ~0.98 kbps compression with high quality reconstruction. The discrete tokens capture enough information to reconstruct intelligible, natural-sounding speech.

Reconstruction is **deterministic** - same codes always produce same audio.
