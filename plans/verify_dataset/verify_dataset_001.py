"""
Script to verify dataset by decoding 5 random samples from a JSONL file.

Usage:
    python nameit_001.py <dataset.jsonl>
"""

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch
from snac import SNAC


def decode_tokens(tokens, model, device):
    """Decode flattened tokens back to audio."""
    # Calculate token sizes for each code level based on stride ratios
    # For strides [8, 4, 2, 1], ratio = 1/8 + 1/4 + 1/2 + 1 = 15/8
    total_tokens = len(tokens)
    ratio = sum(1.0 / s for s in model.vq_strides)
    base_t = int(total_tokens / ratio)
    
    # Unflatten tokens into code levels
    code_levels = []
    start_idx = 0
    for stride in model.vq_strides:
        level_size = base_t // stride
        level_tokens = tokens[start_idx:start_idx + level_size]
        start_idx += level_size
        code_levels.append(torch.tensor(level_tokens, dtype=torch.long, device=device).unsqueeze(0))
    
    # Decode using the model
    with torch.inference_mode():
        audio_reconstructed = model.decode(code_levels)
    
    # Convert to numpy and remove batch/channel dimensions
    return audio_reconstructed.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset by decoding 5 random samples"
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to the JSONL dataset file"
    )
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return
    
    # Get script directory for saving outputs
    script_dir = Path(__file__).parent
    
    # Load SNAC model
    print("Loading SNAC model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    print(f"Using device: {device}")
    
    # Load JSONL file
    print(f"Loading dataset from {jsonl_path}...")
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    # Select 5 random samples
    if len(samples) < 5:
        print(f"Warning: Only {len(samples)} samples available, using all of them")
        selected_samples = samples
    else:
        selected_samples = random.sample(samples, 5)
    
    print(f"\nDecoding {len(selected_samples)} random samples...")
    
    # Decode each sample
    for idx, sample in enumerate(selected_samples, 1):
        text = sample.get('text', 'N/A')
        tokens = sample.get('tokens', [])
        
        if not tokens:
            print(f"Sample {idx}: No tokens found, skipping...")
            continue
        
        print(f"\nSample {idx}:")
        print(f"  Text: {text}")
        print(f"  Tokens: {len(tokens)} tokens")
        
        try:
            # Decode tokens to audio
            audio = decode_tokens(tokens, model, device)
            
            # Save decoded audio
            output_file = script_dir / f"sample_{idx:02d}_decoded.wav"
            sf.write(str(output_file), audio, model.sampling_rate)
            print(f"  ✓ Decoded successfully! Saved to {output_file}")
            
        except Exception as e:
            print(f"  ✗ Error decoding tokens: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Verification complete! Decoded samples saved in {script_dir}")


if __name__ == "__main__":
    main()
