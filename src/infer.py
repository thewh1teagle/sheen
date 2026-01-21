"""Inference for TTS."""
import argparse

import torch
import soundfile as sf
from transformers import AutoModelForCausalLM
from snac import SNAC

from data import deinterleave_snac_tokens, load_tokenizer


def generate_speech(text: str, model_path: str, output_path: str = "output.wav"):
    """Generate speech from phoneme text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    # Encode prompt
    prompt = f"{text}<audio_start>"
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids]).to(device)

    audio_end_id = tokenizer.token_to_id("<audio_end>")
    pad_id = tokenizer.token_to_id("<pad>")

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1000,
            eos_token_id=audio_end_id,
            pad_token_id=pad_id,
            do_sample=False,
        )

    # Extract generated audio tokens
    generated_ids = outputs[0][len(encoded.ids):].tolist()
    tokens = [tokenizer.id_to_token(i) for i in generated_ids if i != audio_end_id]

    # Filter to SNAC tokens only
    snac_tokens = [t for t in tokens if t and t.startswith("<snac_l")]

    if not snac_tokens:
        print("No audio tokens generated")
        return None

    # Decode to audio
    codes = deinterleave_snac_tokens(snac_tokens)
    codes_tensor = [torch.tensor(c).unsqueeze(0).to(device) for c in codes]
    audio = snac.decode(codes_tensor)
    audio_np = audio.squeeze().detach().cpu().numpy()

    sf.write(output_path, audio_np, 24000)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--text", required=True, help="Input phoneme text")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    args = parser.parse_args()

    generate_speech(args.text, args.model, args.output)


if __name__ == "__main__":
    main()
