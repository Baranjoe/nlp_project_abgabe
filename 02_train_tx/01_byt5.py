#!/usr/bin/env python3
"""
Single-GPU ByT5 training script for prepared CSVs.

Unterstützte Varianten (jeweils aus data_prepared/):

    strict, strict_oversampled
    relaxed, relaxed_oversampled

Du steuerst das über:
    DATA_VARIANT = "strict" oder "relaxed"
    USE_OVERSAMPLED = False / True

Die CSVs werden erwartet als:

  strict:
    data_prepared/strict/emb_scores_clean_strict_train.csv
    data_prepared/strict/emb_scores_clean_strict_train_oversampled.csv
    data_prepared/strict/emb_scores_clean_strict_test.csv

  relaxed:
    data_prepared/relaxed/emb_scores_clean_relaxed_train.csv
    data_prepared/relaxed/emb_scores_clean_relaxed_train_oversampled.csv
    data_prepared/relaxed/emb_scores_clean_relaxed_test.csv

Das Modell wird nach:
    models/byt5_<variant>[_oversampled]
gespeichert, z.B.:
    models/byt5_strict
    models/byt5_strict_oversampled
    models/byt5_relaxed
    models/byt5_relaxed_oversampled
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from jiwer import wer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch

# ---------------------------------------------------------------------------
# High-Level Config
# ---------------------------------------------------------------------------

# Welche Datenvariante?
#   "strict"  oder "relaxed"
DATA_VARIANT = "strict"

# Oversampled-Train-Set verwenden?
#   False -> *_train.csv
#   True  -> *_train_oversampled.csv
USE_OVERSAMPLED = False

# Basismodell
MODEL_NAME = "google/byt5-small"

# Ein-/Ausgabe-Basisverzeichnisse
DATA_ROOT = Path("data_prepared")
MODELS_ROOT = Path("models")

# ---------------------------------------------------------------------------
# Training Hyperparams (Single-GPU optimiert)
# ---------------------------------------------------------------------------
MAX_LENGTH = 128

NUM_TRAIN_EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE = 16   # höher als vorher (A100 packt das)
PER_DEVICE_EVAL_BATCH_SIZE = 16

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 200
SAVE_TOTAL_LIMIT = 2
MAX_STEPS = -1  # -1 = volle Epochen, kein Hard-Cut

TRAIN_LIMIT = None   # z.B. 50000 für schnellen Test
VAL_LIMIT = 10000    # Eval-Split bei Bedarf kappen

DATALOADER_NUM_WORKERS = 4  # kannst du auch auf 8 hochdrehen

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_paths(data_variant: str, use_oversampled: bool):
    """
    Erzeuge Pfade für Train/Val-CSV und Output-Modelldir basierend auf Variante.
    """
    if data_variant not in {"strict", "relaxed"}:
        raise ValueError(f"DATA_VARIANT must be 'strict' or 'relaxed', got: {data_variant}")

    base_dir = DATA_ROOT / data_variant

    if data_variant == "strict":
        base_name = "emb_scores_clean_strict"
    else:
        base_name = "emb_scores_clean_relaxed"

    if use_oversampled:
        train_name = f"{base_name}_train_oversampled.csv"
        run_tag_suffix = "_oversampled"
    else:
        train_name = f"{base_name}_train.csv"
        run_tag_suffix = ""

    val_name = f"{base_name}_test.csv"

    train_csv = base_dir / train_name
    val_csv = base_dir / val_name

    run_tag = f"{data_variant}{run_tag_suffix}"
    output_dir = MODELS_ROOT / f"byt5_{run_tag}"

    return train_csv, val_csv, output_dir, run_tag


def preprocess_batch(batch, tokenizer):
    """Tokenize input and target texts for Seq2Seq training."""
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_LENGTH,
        truncation=True,
    )

    # Targets / Labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=MAX_LENGTH,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Compute WER zwischen Dekodierten Predictions und Labels."""
    predictions, labels = eval_pred

    # Manche Modelle liefern (logits, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # HF Seq2SeqTrainer mit predict_with_generate=True gibt
    # meist direkt Token-IDs zurück -> das passt zu batch_decode.
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Ignore-Index -100 durch pad_token_id ersetzen
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wers = [wer(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)]
    return {"wer": float(np.mean(wers))}


def main() -> None:
    # ----------------- Pfade auflösen -----------------
    train_csv, val_csv, output_dir, run_tag = build_paths(DATA_VARIANT, USE_OVERSAMPLED)

    print(f"=== ByT5 Training (Single-GPU) ===")
    print(f"Variante:        {DATA_VARIANT}")
    print(f"Oversampled:     {USE_OVERSAMPLED}")
    print(f"Train CSV:       {train_csv}")
    print(f"Val/Test CSV:    {val_csv}")
    print(f"Output-Dir:      {output_dir}")
    print(f"Base Model:      {MODEL_NAME}")
    print("===================================")

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val/Test CSV not found: {val_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: kleine Metadata-Datei schreiben
    meta = {
        "data_variant": DATA_VARIANT,
        "use_oversampled": USE_OVERSAMPLED,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "model_name": MODEL_NAME,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_length": MAX_LENGTH,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ----------------- Daten laden ------------------
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    required_cols = {"sentence_norm", "transcript_norm"}
    if not required_cols.issubset(df_train.columns) or not required_cols.issubset(df_val.columns):
        raise ValueError(
            f"Both train and val CSV must contain columns {required_cols}, "
            f"but have train={set(df_train.columns)}, val={set(df_val.columns)}"
        )

    # Kürzen, falls gewünscht
    if TRAIN_LIMIT is not None:
        df_train = df_train.iloc[:TRAIN_LIMIT].copy()
    if VAL_LIMIT is not None:
        df_val = df_val.iloc[:VAL_LIMIT].copy()

    print(f"Train-Samples effektiv: {len(df_train)}")
    print(f"Val-Samples effektiv:   {len(df_val)}")

    train_inputs = df_train["transcript_norm"].fillna("").tolist()
    train_targets = df_train["sentence_norm"].fillna("").tolist()

    val_inputs = df_val["transcript_norm"].fillna("").tolist()
    val_targets = df_val["sentence_norm"].fillna("").tolist()

    train_ds = Dataset.from_dict(
        {"input_text": train_inputs, "target_text": train_targets}
    )
    val_ds = Dataset.from_dict(
        {"input_text": val_inputs, "target_text": val_targets}
    )

    # ----------------- Tokenizer & Modell ------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # optional kleine Single-GPU Optimierung
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No CUDA device found – training will run on CPU.")

    # ----------------- Tokenization -----------------------
    train_tokenized = train_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ----------------- Training args (Single-GPU) ----------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # Deine HF-Version nutzt offenbar 'eval_strategy'
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        predict_with_generate=True,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,  # -1 = volle Epochen

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,

        logging_steps=LOGGING_STEPS,
        report_to="none",

        bf16=True,                      # A100 kann das
        gradient_checkpointing=False,   # für Speed erstmal aus; bei OOM -> True setzen
        ddp_find_unused_parameters=False,

        dataloader_num_workers=DATALOADER_NUM_WORKERS,
    )

    # ----------------- Trainer ----------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda batch: compute_metrics(batch, tokenizer),
    )

    # ----------------- Train & save -----------------------
    trainer.train()
    trainer.save_state()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
