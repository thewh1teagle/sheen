"""Dataset and SNAC token formatting."""
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import SPECIAL_TOKENS, SNAC_TOKENS


def setup_tokenizer(tokenizer_model):
    """Load Qwen tokenizer and add SNAC tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, fix_mistral_regex=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.add_tokens(SNAC_TOKENS)

    # Set pad token for batched training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


@dataclass
class TTSDataCollator:
    """
    Data collator for TTS training with custom label masking.

    Pads input_ids and attention_mask to the longest sequence in the batch.
    Labels are padded with -100 (CrossEntropyLoss ignore index) so padded
    positions don't contribute to loss.

    Note: Truncation should be handled by the dataset, not the collator.
    """

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Extract labels before padding (tokenizer.pad doesn't handle them)
        labels = [f.pop("labels") for f in features]

        # Pad input_ids and attention_mask to longest in batch
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # Pad labels with -100 so padding doesn't affect loss
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_length = max_length - len(label)
            padded_labels.append(label + [self.label_pad_token_id] * pad_length)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


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

    Format: {text}<audio_start>{audio_tokens}<audio_end>

    SNAC tokens are concatenated without spaces since they're discrete tokens
    added to the vocabulary - no whitespace needed between them.
    """
    audio_tokens = interleave_snac_codes(snac_codes)
    return f"{text}<audio_start>{''.join(audio_tokens)}<audio_end>"


class TTSDataset(Dataset):
    """
    Dataset for TTS training.

    Creates samples where the model learns to predict audio tokens given text.
    Text tokens are masked from the loss computation - only audio token
    predictions contribute to training.
    """

    LABEL_IGNORE_INDEX = -100

    def __init__(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Cache special token IDs for efficient label creation
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")
        if self.audio_start_id is None:
            raise ValueError("Tokenizer missing <audio_start> token. Run setup_tokenizer() first.")

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

        input_ids = encoded["input_ids"]
        encoded["labels"] = self._create_labels(input_ids)

        return encoded

    def _create_labels(self, input_ids):
        """
        Create labels that only compute loss on audio token predictions.

        In causal LM, at position i the model predicts token i+1. We want:
        - NO loss for predicting text tokens (irrelevant to TTS task)
        - NO loss for predicting <audio_start> (it's a fixed delimiter)
        - Loss for predicting audio tokens and <audio_end>

        This is achieved by setting labels to -100 (ignore index) for all
        positions where we don't want loss computed.

        Args:
            input_ids: Token IDs in format [text..., <audio_start>, audio..., <audio_end>]

        Returns:
            Labels with text positions masked, audio positions preserved.
        """
        try:
            audio_start_pos = input_ids.index(self.audio_start_id)
        except ValueError:
            # <audio_start> not found - likely truncated. Mask entire sequence.
            return [self.LABEL_IGNORE_INDEX] * len(input_ids)

        # Mask: text tokens + <audio_start>
        # Keep: audio tokens + <audio_end>
        labels = [self.LABEL_IGNORE_INDEX] * (audio_start_pos + 1)
        labels += input_ids[audio_start_pos + 1:]

        return labels


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
