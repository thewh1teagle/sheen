# Sheen repo review (2026-01-21)

## Scope
Reviewed training, data prep, dataset/labeling, inference, and config.
Files: `src/config.py`, `src/data.py`, `src/train.py`, `src/infer.py`, `scripts/prepare_dataset.py`, `README.md`.

## Architecture summary
- Model: base LLM (default `Qwen/Qwen3-0.6B`) fine-tuned as a causal LM to emit SNAC audio tokens.
- Input format: `{text}<audio_start>{snac_tokens}<audio_end>`; loss masked to only audio tokens.
- Tokenization: adds `<audio_start>/<audio_end>` plus 3*4096 SNAC tokens to vocab.
- Data: JSONL with `text` and `snac_codes` (3 layers). `prepare_dataset.py` builds JSONL from `metadata.csv` + wav.
- Inference: greedy generate to `<audio_end>`, filter SNAC tokens, decode with SNAC codec.

## Strengths
- Clear minimal pipeline: dataset prep -> tokenizer -> Trainer -> inference.
- Label masking isolates audio-token prediction, matching intended task.
- Deterministic train/eval split with seed.
- Decoding flow is straightforward and uses explicit `<audio_end>` stop.

## Findings and risks (ranked)
1) Dataset truncation silently drops audio tail
   - `TTSDataset` truncates to `max_length` before labeling; if `<audio_start>` or audio tokens are truncated, labels are fully masked or audio tail is cut. This can reduce effective training signal without warning. Consider adding stats/logging for truncation rate and/or pre-truncating audio tokens to fit text.

2) `Trainer` config may not evaluate if eval split is 0 samples
   - Small datasets + `eval_split=0.05` can lead to `eval_size=0`, but `eval_strategy="steps"` and `eval_steps` still set. This can error or behave unexpectedly. Add guard to disable eval or require min eval size.

3) Tokenizer + model resizing without tying / init check
   - `model.resize_token_embeddings(len(tokenizer))` is correct, but no explicit init for new SNAC tokens. Defaults to random init; that is expected, but consider logging or initializing from a distribution matched to existing embeddings if instability appears.

4) Collator assumes labels already aligned to input_ids
   - It pads labels with -100, but no validation that label length matches input_ids length. This is fine now, but if future preprocessing adds anything, it can silently misalign. Add an assert for length equality.

5) Inference prompt has a space before `<audio_start>`
   - `prompt = f"{text} <audio_start>"` inserts a literal space before the token. In training, format is `{text}<audio_start>` with no space. This mismatch could slightly hurt generation. Align formats.

6) Missing safety checks in dataset prep
   - `prepare_dataset.py` skips missing audio but does not log counts or enforce minimum length. It also resamples with `signal.resample` (FFT) which can introduce artifacts; consider `resampy` or `torchaudio` resampler for quality.

## Training readiness notes
- Memory: adding 12,288 SNAC tokens expands embeddings; for 0.6B model this is manageable but still adds overhead. Watch GPU memory when increasing `max_length`.
- Loss masking: only audio tokens contribute, which is correct for conditional generation, but consider also predicting `<audio_end>` (currently included) to stabilize stopping.
- `processing_class=tokenizer` in `Trainer` is for HF 4.44+; if pinned older, this may break. Check `transformers` version in `pyproject.toml` and update if needed.

## Suggested verification (optional)
- Quick dry-run with a tiny JSONL and `--max-length` small to confirm:
  - labels are masked correctly (text tokens ignored, audio tokens kept)
  - `eval_steps` works with chosen eval split
- Run `uv run scripts/prepare_dataset.py --dataset-dir dataset/` on a small subset to verify SNAC encode pipeline.

## Quick wins
- Align inference prompt formatting to training format.
- Add logging for truncation rate and sample lengths.
- Add a minimum eval size check or set `eval_strategy="no"` when eval size is 0.
- Add asserts in collator for label length.

