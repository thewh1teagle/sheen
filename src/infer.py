"""Inference for TTS with delay pattern."""
import argparse

import torch
import soundfile as sf
from transformers import AutoModelForCausalLM
from snac import SNAC

from config import DEFAULT_DELAY_L2, DEFAULT_DELAY_L3
from data import load_tokenizer

# Tokens per frame: 1 L1 + 2 L2 + 4 L3
TOKENS_PER_FRAME = 7


def generate_speech(
    text: str,
    model_path: str,
    output_path: str = "output.wav",
    max_frames: int = 150,
):
    """Generate speech from phoneme text using delay pattern."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    codes = _generate_with_delay(text, model, tokenizer, device, max_frames)

    if codes is None:
        print("No audio tokens generated")
        return None

    # Decode to audio
    codes_tensor = [torch.tensor(c).unsqueeze(0).to(device) for c in codes]
    audio = snac.decode(codes_tensor)
    audio_np = audio.squeeze().detach().cpu().numpy()

    sf.write(output_path, audio_np, 24000)
    print(f"Saved: {output_path}")
    return output_path


def _generate_with_delay(
    text: str,
    model,
    tokenizer,
    device: str,
    max_frames: int,
) -> list[list[int]] | None:
    """
    Generate with delay pattern matching training.

    During training, the model saw delayed context for L2/L3 codebooks.
    This generation loop maintains that pattern for consistent behavior.
    """
    audio_end_id = tokenizer.token_to_id("<audio_end>")
    mask_id = tokenizer.token_to_id("<audio_mask>")

    prompt = f"{text}<audio_start>"
    prompt_ids = tokenizer.encode(prompt).ids

    # Storage for generated tokens (original alignment)
    l1_ids: list[int] = []
    l2_ids: list[int] = []
    l3_ids: list[int] = []

    for frame in range(max_frames):
        # Build context with delay pattern applied
        context_ids = _build_delayed_context(
            prompt_ids, l1_ids, l2_ids, l3_ids, frame, mask_id
        )

        # Generate 7 tokens for this frame
        frame_tokens = _generate_frame_tokens(
            model, context_ids, device, audio_end_id, mask_id
        )

        if frame_tokens is None:
            break

        # Store in original alignment
        l1_ids.append(frame_tokens[0])
        l2_ids.extend(frame_tokens[1:3])
        l3_ids.extend(frame_tokens[3:7])

    if not l1_ids:
        return None

    # Convert token IDs to SNAC codes
    l1_codes = [_token_id_to_code(tokenizer, tid) for tid in l1_ids]
    l2_codes = [_token_id_to_code(tokenizer, tid) for tid in l2_ids]
    l3_codes = [_token_id_to_code(tokenizer, tid) for tid in l3_ids]

    # Filter out any failed conversions
    if None in l1_codes or None in l2_codes or None in l3_codes:
        return None

    return [l1_codes, l2_codes, l3_codes]


def _build_delayed_context(
    prompt_ids: list[int],
    l1_ids: list[int],
    l2_ids: list[int],
    l3_ids: list[int],
    current_frame: int,
    mask_id: int,
) -> list[int]:
    """Build input context with delay pattern for frames 0..current_frame-1."""
    context = list(prompt_ids)

    for f in range(current_frame):
        # L1: no delay
        context.append(l1_ids[f])

        # L2: delayed by DEFAULT_DELAY_L2 frames
        l2_frame = f - DEFAULT_DELAY_L2
        for j in range(2):
            l2_idx = l2_frame * 2 + j
            if l2_frame >= 0 and l2_idx < len(l2_ids):
                context.append(l2_ids[l2_idx])
            else:
                context.append(mask_id)

        # L3: delayed by DEFAULT_DELAY_L3 frames
        l3_frame = f - DEFAULT_DELAY_L3
        for j in range(4):
            l3_idx = l3_frame * 4 + j
            if l3_frame >= 0 and l3_idx < len(l3_ids):
                context.append(l3_ids[l3_idx])
            else:
                context.append(mask_id)

    return context


def _generate_frame_tokens(
    model,
    context_ids: list[int],
    device: str,
    audio_end_id: int,
    mask_id: int,
) -> list[int] | None:
    """Generate 7 tokens for one frame."""
    context = list(context_ids)
    frame_tokens = []

    with torch.inference_mode():
        for pos in range(TOKENS_PER_FRAME):
            input_tensor = torch.tensor([context]).to(device)
            output = model(input_tensor)
            next_token_id = output.logits[0, -1].argmax().item()

            if next_token_id == audio_end_id:
                return None

            frame_tokens.append(next_token_id)

            # Add to context with delay (use mask for same-frame L2/L3)
            if pos == 0:
                # L1 has no delay, add directly
                context.append(next_token_id)
            else:
                # L2/L3 within same frame: use mask (delay not yet elapsed)
                context.append(mask_id)

    return frame_tokens


def _token_id_to_code(tokenizer, token_id: int) -> int | None:
    """Extract SNAC code from token ID."""
    token = tokenizer.id_to_token(token_id)
    if not token or not token.startswith("<snac_l"):
        return None
    try:
        return int(token.split("_")[2].rstrip(">"))
    except (IndexError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Input phoneme text")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum frames to generate (default: 150, ~12 sec)",
    )
    args = parser.parse_args()

    generate_speech(args.text, args.model, args.output, args.max_frames)


if __name__ == "__main__":
    main()
