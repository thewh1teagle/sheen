# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.21.1", "requests==2.32.3"]
# ///
"""
Build a character-level tokenizer for TTS.

Usage: uv run scripts/build_tokenizer.py tokenizer.json
"""
import argparse

import requests
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

SNAC_TOKENS = [f"<snac_l{level}_{code}>" for level in range(1, 4) for code in range(4096)]
PIPER_URL = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ljspeech/medium/config.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output tokenizer.json path")
    args = parser.parse_args()

    # Get phonemes from Piper
    phonemes = list(requests.get(PIPER_URL).json()["phoneme_id_map"].keys())
    print(f"Phonemes: {len(phonemes)}")

    # Build vocab
    vocab = {"<unk>": 0}
    for p in sorted(phonemes):
        if p not in vocab:
            vocab[p] = len(vocab)

    # Create tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

    # Add special tokens (matched before pre-tokenizer splits)
    special = ["<pad>", "<audio_start>", "<audio_end>"] + SNAC_TOKENS
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in special])

    # Test
    print(f"Vocab: {tokenizer.get_vocab_size()}")
    test = "bˈajit<audio_start><snac_l1_100><audio_end>"
    print(f"Test: {tokenizer.encode(test).tokens}")

    tokenizer.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
