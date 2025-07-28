# learn.py
"""
Called by Streamlit 'Fine‑tune' button or as: python -m learn
Produces models/go‑emotions‑lora/
"""
import torch
from pathlib import Path
from db import get_connection
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

BASE = "AnasAlokla/multilingual_go_emotions_V1.1"
DB = "girlfriend.db"
OUT = Path("models/go-emotions-lora")

# Ensure output directory exists
OUT.parent.mkdir(parents=True, exist_ok=True)


def dataset_from_feedback():
    con = get_connection()
    rows = con.execute(
        """
        SELECT text, emotion FROM msgs
        WHERE id IN (
            SELECT text_id FROM feedback WHERE res = 'it_is'
        )
        """
    ).fetchall()
    con.close()

    texts, labels = zip(*rows) if rows else ([], [])
    return list(texts), list(labels)


def run_finetune():
    # DEBUG: how many confirmed examples?
    texts, labels = dataset_from_feedback()
    print(f"[learn.py] 🔍 Found {len(texts)} ‘it_is’ examples for fine‑tuning")
    # Backup old adapter if exists
    if OUT.exists():
        OUT.rename(OUT.with_suffix(".bak"))

    if not texts:
        print("No feedback found, skipping fine-tune.")
        return

    # Tokenize
    tok = AutoTokenizer.from_pretrained(BASE)
    enc = tok(texts, truncation=True, padding=True, return_tensors='pt')

    # Convert labels to token IDs
    lbl = torch.tensor([tok.convert_tokens_to_ids([l])[0] for l in labels])

    # Build dataset
    ds = torch.utils.data.TensorDataset(enc['input_ids'], enc['attention_mask'], lbl)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=tok.vocab_size
    )
    # Apply LoRA
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"])
    model = get_peft_model(model, peft_cfg)

    # Training arguments
    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=3e-4,
        logging_steps=10,
        save_strategy="no"
    )

    # Train
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    # Save adapter and tokenizer
    model.save_pretrained(str(OUT))  # type: ignore
    tok.save_pretrained(str(OUT))    # type: ignore


if __name__ == "__main__":
    run_finetune()
