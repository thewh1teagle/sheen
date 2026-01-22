"""Dataset and data utilities for TTS training."""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from config import DEFAULT_DELAY_L1, DEFAULT_DELAY_L2, DEFAULT_DELAY_L3

TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"
MASK_TOKEN = "<audio_mask>"


def load_tokenizer(path: str | Path = TOKENIZER_PATH) -> Tokenizer:
    """Load the character-level tokenizer."""
    return Tokenizer.from_file(str(path))


def interleave_snac_codes(snac_codes: list[list[int]]) -> list[str]:
    """
    Convert SNAC codes to frame-interleaved token strings.
    Each frame: 1 L1 + 2 L2 + 4 L3 = 7 tokens
    """
    l1, l2, l3 = snac_codes
    tokens = []

    for i in range(len(l1)):
        if i * 2 + 1 >= len(l2) or i * 4 + 3 >= len(l3):
            break

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


def interleave_snac_codes_with_delay(
    snac_codes: list[list[int]],
    delay_l1: int = DEFAULT_DELAY_L1,
    delay_l2: int = DEFAULT_DELAY_L2,
    delay_l3: int = DEFAULT_DELAY_L3,
) -> tuple[list[str], list[str]]:
    """
    Convert SNAC codes to frame-interleaved tokens with delay pattern.

    The delay pattern staggers codebook visibility during training:
    - L1 sees current frame (delay=0 by default)
    - L2 sees previous frame (delay=1 by default)
    - L3 sees two frames back (delay=2 by default)

    This prevents the model from "cheating" by copying adjacent codebook
    values and forces it to learn the actual acoustic structure.

    Returns:
        input_tokens: Delayed sequence (model input during forward pass)
        label_tokens: Original sequence (prediction targets for loss)
    """
    l1, l2, l3 = snac_codes
    input_tokens = []
    label_tokens = []

    num_frames = len(l1)
    for i in range(num_frames):
        if i * 2 + 1 >= len(l2) or i * 4 + 3 >= len(l3):
            break

        # L1 tokens
        l1_delayed_frame = i - delay_l1
        if l1_delayed_frame >= 0:
            input_tokens.append(f"<snac_l1_{l1[l1_delayed_frame]}>")
        else:
            input_tokens.append(MASK_TOKEN)
        label_tokens.append(f"<snac_l1_{l1[i]}>")

        # L2 tokens (2 per frame)
        l2_delayed_frame = i - delay_l2
        for j in range(2):
            l2_idx = l2_delayed_frame * 2 + j
            if l2_delayed_frame >= 0 and l2_idx < len(l2):
                input_tokens.append(f"<snac_l2_{l2[l2_idx]}>")
            else:
                input_tokens.append(MASK_TOKEN)
            label_tokens.append(f"<snac_l2_{l2[i * 2 + j]}>")

        # L3 tokens (4 per frame)
        l3_delayed_frame = i - delay_l3
        for j in range(4):
            l3_idx = l3_delayed_frame * 4 + j
            if l3_delayed_frame >= 0 and l3_idx < len(l3):
                input_tokens.append(f"<snac_l3_{l3[l3_idx]}>")
            else:
                input_tokens.append(MASK_TOKEN)
            label_tokens.append(f"<snac_l3_{l3[i * 4 + j]}>")

    return input_tokens, label_tokens


def deinterleave_snac_tokens(tokens: list[str]) -> list[list[int]]:
    """Convert token strings back to SNAC codes. Ignores mask tokens."""
    l1, l2, l3 = [], [], []

    for token in tokens:
        if not token.startswith("<snac_l"):
            continue
        parts = token[1:-1].split("_")
        layer = int(parts[1][1])
        code = int(parts[2])

        if layer == 1:
            l1.append(code)
        elif layer == 2:
            l2.append(code)
        else:
            l3.append(code)

    return [l1, l2, l3]


class TTSDataset(Dataset):
    """Dataset for TTS training with delay pattern."""

    LABEL_IGNORE_INDEX = -100

    def __init__(self, path: str, tokenizer: Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.audio_start_id = tokenizer.token_to_id("<audio_start>")
        self.audio_end_id = tokenizer.token_to_id("<audio_end>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        snac_codes = sample["snac_codes"]

        input_tokens, label_tokens = interleave_snac_codes_with_delay(snac_codes)

        input_text = f"{text}<audio_start>{''.join(input_tokens)}<audio_end>"
        label_text = f"{text}<audio_start>{''.join(label_tokens)}<audio_end>"

        input_encoded = self.tokenizer.encode(input_text)
        label_encoded = self.tokenizer.encode(label_text)

        input_ids = input_encoded.ids
        label_ids = label_encoded.ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            label_ids = label_ids[: self.max_length]

        labels = self._create_labels(label_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _create_labels(self, label_ids: list[int]) -> list[int]:
        """Mask text tokens, keep original audio tokens for loss."""
        try:
            audio_start_pos = label_ids.index(self.audio_start_id)
        except ValueError:
            return [self.LABEL_IGNORE_INDEX] * len(label_ids)

        labels = [self.LABEL_IGNORE_INDEX] * (audio_start_pos + 1)
        labels += label_ids[audio_start_pos + 1 :]
        return labels


class TTSDataCollator:
    """Collator that pads batches and handles labels."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
