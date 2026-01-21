"""
wget <dataset.7z>
7z x heb-female-audio-ipa2-v2.7z 
mv heb-female-audio-ipa2-v2 dataset/

uv run scripts/prepare_dataset.py --dataset-dir dataset/
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import soundfile as sf
from scipy import signal
from snac import SNAC
from tqdm import tqdm


def load_and_preprocess_audio(audio_file: Path, target_sr: int = 24000):
    """Load audio file and preprocess it (mono conversion, resampling)."""
    audio, sr = sf.read(str(audio_file), dtype='float32')
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to target sample rate if necessary
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)
    
    return audio


def encode_audio(audio, model, device):
    """Encode audio with SNAC and return codes structured by layer."""
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        codes = model.encode(audio_tensor)
    
    # Keep codes structured by layer (layer1, layer2, layer3)
    # Each layer is a list of integers representing SNAC codes
    layer_codes = []
    for code_level in codes:
        layer_codes.append(code_level.squeeze(0).cpu().tolist())
    
    return layer_codes


def process_entry(row, dataset_dir: Path, model, device):
    """Process a single metadata entry."""
    audio_id = str(row['id']).strip()
    phonemes = str(row['phonemes']).strip()
    
    # Find audio file (.wav only)
    audio_file = dataset_dir / 'wav'/ f"{audio_id}.wav"
    if not audio_file.exists():
        return None, f"Audio file not found for ID {audio_id}"
    
    try:
        audio = load_and_preprocess_audio(audio_file)
        snac_codes = encode_audio(audio, model, device)
        
        return {
            "text": phonemes,
            "snac_codes": snac_codes
        }, None
    except Exception as e:
        return None, f"Error processing {audio_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from metadata.csv")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/",
        help="Directory containing metadata.csv (default: dataset/)"
    )
    args = parser.parse_args()
    
    # Load SNAC model
    print("Loading SNAC model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    print(f"Using device: {device}")
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    metadata_file = dataset_dir / "metadata.csv"
    output_file = dataset_dir / "dataset.jsonl"
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_file, delimiter='|', header=None, names=['id', 'phonemes'])
    
    # Process audio files
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
            result, error = process_entry(row, dataset_dir, model, device)
            
            if error:
                tqdm.write(f"Warning: {error}, skipping...")
                continue
            
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Done! Created {output_file}")


if __name__ == "__main__":
    main()

