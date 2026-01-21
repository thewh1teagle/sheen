"""Dataset and SNAC token formatting."""
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import SPECIAL_TOKENS, SNAC_TOKENS


def setup_tokenizer(model_name):
    """Load Qwen tokenizer and add SNAC tokens."""
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
