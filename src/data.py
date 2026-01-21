"""Dataset and data utilities for TTS training."""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"


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


def deinterleave_snac_tokens(tokens: list[str]) -> list[list[int]]:
    """Convert token strings back to SNAC codes."""
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
    """Dataset for TTS training."""

    LABEL_IGNORE_INDEX = -100

    def __init__(self, path: str, tokenizer: Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.audio_start_id = tokenizer.token_to_id("<audio_start>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        with open(path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        audio_tokens = interleave_snac_codes(sample["snac_codes"])

        # Format: {text}<audio_start>{audio_tokens}<audio_end>
        full_text = f"{text}<audio_start>{''.join(audio_tokens)}<audio_end>"

        # Encode
        encoded = self.tokenizer.encode(full_text)
        input_ids = encoded.ids

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Create labels (mask text, keep audio)
        labels = self._create_labels(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _create_labels(self, input_ids: list[int]) -> list[int]:
        """Mask text tokens, keep audio tokens for loss."""
        try:
            audio_start_pos = input_ids.index(self.audio_start_id)
        except ValueError:
            return [self.LABEL_IGNORE_INDEX] * len(input_ids)

        labels = [self.LABEL_IGNORE_INDEX] * (audio_start_pos + 1)
        labels += input_ids[audio_start_pos + 1:]
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
