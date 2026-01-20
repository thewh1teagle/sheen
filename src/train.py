from transformers import Qwen3Model, Qwen3Config

text_vocab_size = 151936
audio_vocab_size = 1024

model = Qwen3Model(Qwen3Config(
    hidden_size=512,
    intermediate_size=512*3,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=128,
    max_position_embeddings=2048,
    vocab_size=text_vocab_size + audio_vocab_size
))
model.eval()
model.to("cuda")