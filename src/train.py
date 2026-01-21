"""Training script for Qwen+SNAC TTS."""
import random

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from config import get_args
from data import setup_tokenizer, TTSDataset, TTSDataCollator


def main():
    args = get_args()

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    # Tokenizer
    tokenizer = setup_tokenizer(args.model)
    print(f"Vocab size: {len(tokenizer)}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Dataset with shuffled train/eval split
    dataset = TTSDataset(args.dataset, tokenizer, args.max_length)

    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)

    eval_size = int(len(dataset) * args.eval_split)
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)} (split={args.eval_split}, seed={args.seed})")

    # Data collator for batched training with proper padding
    data_collator = TTSDataCollator(tokenizer=tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"Saved to {args.output}/final")


if __name__ == "__main__":
    main()
