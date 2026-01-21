"""Model configuration and creation."""
from transformers import Qwen3Config, Qwen3ForCausalLM

from config import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, MAX_LENGTH


def create_model(vocab_size):
    """
    Create a custom Qwen3 model from scratch.
    
    Args:
        vocab_size: Total vocabulary size (text + audio tokens)
    
    Returns:
        Qwen3ForCausalLM model
    """
    config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_LENGTH,
        vocab_size=vocab_size,
    )
    model = Qwen3ForCausalLM(config)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("Model Architecture: Qwen3 (custom)")
    print(f"{'='*60}")
    print(f"Layers:          {config.num_hidden_layers}")
    print(f"Hidden size:     {config.hidden_size}")
    print(f"Intermediate:    {config.intermediate_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"KV heads:        {config.num_key_value_heads}")
    print(f"Vocab size:      {config.vocab_size:,}")
    print(f"Max length:      {config.max_position_embeddings}")
    print(f"Total params:    {total_params/1e6:.2f}M")
    print(f"Trainable:       {trainable_params/1e6:.2f}M")
    print(f"{'='*60}\n")
    
    return model
