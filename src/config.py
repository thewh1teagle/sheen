"""Training configuration."""
import argparse

# SNAC constants
SNAC_VOCAB_SIZE = 4096
SNAC_LAYERS = 3
SNAC_SAMPLE_RATE = 24000
SNAC_TOKENS = [f"<snac_l{l}_{c}>" for l in [1, 2, 3] for c in range(SNAC_VOCAB_SIZE)]
SPECIAL_TOKENS = ["<audio_start>", "<audio_end>"]


def get_args():
    parser = argparse.ArgumentParser(description="Train Qwen+SNAC TTS")

    # Required
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model")
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
