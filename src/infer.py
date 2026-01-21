"""Inference for Qwen+SNAC TTS."""
import argparse

import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

from data import deinterleave_snac_tokens


def generate_speech(text, model_path, output_path="output.wav"):
    """Generate speech from text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    # Generate
    prompt = f"{text}<audio_start>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            eos_token_id=audio_end_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    # Extract audio tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]  # Skip prompt
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    # Filter to just SNAC tokens (before <audio_end>)
    snac_tokens = []
    for t in tokens:
        if t == "<audio_end>":
            break
        if t.startswith("<snac_l"):
            snac_tokens.append(t)

    # Convert back to codes
    codes = deinterleave_snac_tokens(snac_tokens)

    # Decode audio
    codes_tensor = [torch.tensor(c).unsqueeze(0).to(device) for c in codes]
    audio = snac.decode(codes_tensor)
    audio_np = audio.squeeze().cpu().numpy()

    sf.write(output_path, audio_np, 24000)
    print(f"Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Input text/phonemes")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    args = parser.parse_args()

    generate_speech(args.text, args.model, args.output)


if __name__ == "__main__":
    main()
