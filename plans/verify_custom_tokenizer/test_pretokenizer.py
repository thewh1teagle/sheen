# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.21.1"]
# ///
"""Test different pre-tokenizer configurations."""
from tokenizers import pre_tokenizers

test = "hello<audio_start><snac_l1_100>world"

# Test different patterns and configurations
configs = [
    # Try empty string split (char by char)
    ("Split '' isolated", pre_tokenizers.Split("", "isolated", False)),

    # Try different regex patterns
    ("Pattern: <.*?>", pre_tokenizers.Split(r"<.*?>", "isolated", False)),
    ("Pattern: <[^>]+>", pre_tokenizers.Split(r"<[^>]+>", "isolated", False)),

    # Sequence approach
    ("Sequence(Whitespace)", pre_tokenizers.Whitespace()),

    # Try CharDelimiterSplit
    ("CharDelimiterSplit(' ')", pre_tokenizers.CharDelimiterSplit(' ')),
]

print(f"Input: '{test}'\n")
for name, pt in configs:
    try:
        result = pt.pre_tokenize_str(test)
        tokens = [t[0] for t in result]
        print(f"{name}:")
        print(f"  {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print()
    except Exception as e:
        print(f"{name}: ERROR - {e}\n")

# Test: What if we just don't use pre-tokenizer and add all chars as vocab?
print("=" * 40)
print("Alternative: No pre-tokenizer, just use vocab lookup")
print("This requires a custom encode function")

