"""Training script for Qwen+SNAC TTS."""
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from config import get_args
from data import setup_tokenizer, TTSDataset


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

    # Dataset
    dataset = TTSDataset(args.dataset, tokenizer, args.max_length)
    split_idx = int(len(dataset) * 0.9)
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"Saved to {args.output}/final")


if __name__ == "__main__":
    main()
