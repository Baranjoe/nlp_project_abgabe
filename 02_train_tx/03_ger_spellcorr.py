#!/usr/bin/env python3
"""
Single-GPU Seq2Seq Training Script for mT5 / German Spelling Correction

Verwendet den vorbereiteten CSVs in data_prepared:

  strict:
    - data_prepared/strict/emb_scores_clean_strict_train.csv
    - data_prepared/strict/emb_scores_clean_strict_train_oversampled.csv

  relaxed:
    - data_prepared/relaxed/emb_scores_clean_relaxed_train.csv
    - data_prepared/relaxed/emb_scores_clean_relaxed_train_oversampled.csv

Konfiguration:
  - DATA_VARIANT  : 'strict', 'strict_oversampled', 'relaxed', 'relaxed_oversampled'
  - MODEL_VARIANT : 'mt5-small', 'ger-spellcorr-base'

Beispiele:
  - mT5 auf strict:              DATA_VARIANT='strict', MODEL_VARIANT='mt5-small'
  - mT5 auf strict+oversampled:  DATA_VARIANT='strict_oversampled', MODEL_VARIANT='mt5-small'
  - German-Speller auf relaxed:  DATA_VARIANT='relaxed', MODEL_VARIANT='ger-spellcorr-base'
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from datasets import Dataset
from jiwer import wer
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------------------------
# High-level config: HIER ANPASSEN
# ---------------------------------------------------------------------------

# Welche Datenvariante soll für das Training genutzt werden?
#   'strict'
#   'strict_oversampled'
#   'relaxed'
#   'relaxed_oversampled'
DATA_VARIANT = "strict"

# Welches Basismodell?
#   'mt5-small'           -> google/mt5-small
#   'ger-spellcorr-base'  -> oliverguhr/spelling-correction-german-base
MODEL_VARIANT = "ger-spellcorr-base"

# ---------------------------------------------------------------------------
# Derived paths / model names (normalerweise NICHT ändern)
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data_prepared")

DATA_PATHS = {
    "strict": DATA_ROOT / "strict" / "emb_scores_clean_strict_train.csv",
    "strict_oversampled": DATA_ROOT / "strict" / "emb_scores_clean_strict_train_oversampled.csv",
    "relaxed": DATA_ROOT / "relaxed" / "emb_scores_clean_relaxed_train.csv",
    "relaxed_oversampled": DATA_ROOT / "relaxed" / "emb_scores_clean_relaxed_train_oversampled.csv",
}

MODEL_MAP = {
    "mt5-small": "google/mt5-small",
    "ger-spellcorr-base": "oliverguhr/spelling-correction-german-base",
}

if DATA_VARIANT not in DATA_PATHS:
    raise ValueError(f"Unknown DATA_VARIANT '{DATA_VARIANT}'. Expected one of {list(DATA_PATHS.keys())}")

if MODEL_VARIANT not in MODEL_MAP:
    raise ValueError(f"Unknown MODEL_VARIANT '{MODEL_VARIANT}'. Expected one of {list(MODEL_MAP.keys())}")

INPUT_CSV = DATA_PATHS[DATA_VARIANT]
MODEL_NAME = MODEL_MAP[MODEL_VARIANT]

# Output-Verzeichnis:
# z.B. models/mt5-small_strict  oder  models/ger-spellcorr-base_relaxed_oversampled
OUTPUT_ROOT = Path("models")
RUN_NAME = f"{MODEL_VARIANT}_{DATA_VARIANT}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

VAL_FRACTION = 0.1        # Anteil des Train-CVS als Validierung
TRAIN_LIMIT = None        # z.B. 100_000 zum Debuggen, sonst None
VAL_LIMIT = None          # optionales Limit für Val

MAX_LENGTH = 128
NUM_TRAIN_EPOCHS = 3.0
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2
MAX_STEPS = -1            # -1 = volle Epochen

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def preprocess_batch(batch, tokenizer):
    """Tokenize input and target texts for Seq2Seq training."""
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_LENGTH,
        truncation=True,
    )

    # Target / labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=MAX_LENGTH,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """
    Compute WER between decoded predictions and labels.
    Robust gegenüber verschiedenen prediction-Formaten (Logits vs Token-IDs).
    """
    predictions, labels = eval_pred

    # Manche Modelle liefern (logits, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # FALL 1: predictions sind Logits -> (batch, seq_len, vocab)
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    # Jetzt (batch, seq_len) mit int-ähnlichen Werten
    predictions = predictions.astype(np.int32)
    labels = labels.astype(np.int32)

    # Labels: -100 -> pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # Safety-Clip in den gültigen Vokab-Bereich
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        predictions = np.clip(predictions, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wers = [wer(ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)]
    return {"wer": float(np.mean(wers))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ----------------- I/O checks -----------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Training config ===")
    print(f"  DATA_VARIANT  = {DATA_VARIANT}")
    print(f"  MODEL_VARIANT = {MODEL_VARIANT}")
    print(f"  INPUT_CSV     = {INPUT_CSV}")
    print(f"  OUTPUT_DIR    = {OUTPUT_DIR}")
    print("=======================\n")

    # ----------------- Load data ------------------
    df = pd.read_csv(INPUT_CSV)
    required_cols = {"sentence_norm", "transcript_norm"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns {required_cols}, "
            f"but has {set(df.columns)}"
        )

    # Optional: Trainingslimit zum Debuggen
    if TRAIN_LIMIT is not None and len(df) > TRAIN_LIMIT:
        df = df.sample(TRAIN_LIMIT, random_state=42).reset_index(drop=True)
        print(f"Applied TRAIN_LIMIT={TRAIN_LIMIT}, new len(df)={len(df)}")

    inputs = df["transcript_norm"].fillna("").tolist()
    targets = df["sentence_norm"].fillna("").tolist()

    indices = np.arange(len(df))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_FRACTION,
        random_state=42,
        shuffle=True,
    )

    train_inputs = [inputs[i] for i in train_indices]
    val_inputs = [inputs[i] for i in val_indices]
    train_targets = [targets[i] for i in train_indices]
    val_targets = [targets[i] for i in val_indices]

    print(f"Train size: {len(train_inputs)}, Val size: {len(val_inputs)}")

    train_ds = Dataset.from_dict(
        {"input_text": train_inputs, "target_text": train_targets}
    )
    val_ds = Dataset.from_dict(
        {"input_text": val_inputs, "target_text": val_targets}
    )

    if VAL_LIMIT is not None and len(val_ds) > VAL_LIMIT:
        val_ds = val_ds.select(range(VAL_LIMIT))
        print(f"Applied VAL_LIMIT={VAL_LIMIT}, new len(val_ds)={len(val_ds)}")

    # ----------------- Tokenizer & model ------------------
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ----------------- Tokenization -----------------------
    print("Tokenizing datasets...")
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

    # ----------------- Training args ----------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

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

        # Single-GPU A100-Feintuning
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,  # wird im Single-GPU-Setup ignoriert
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

    # ----------------- Save config meta -------------------
    meta = {
        "data_variant": DATA_VARIANT,
        "model_variant": MODEL_VARIANT,
        "model_name": MODEL_NAME,
        "input_csv": str(INPUT_CSV),
        "val_fraction": VAL_FRACTION,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "train_size": len(train_inputs),
        "val_size": len(val_inputs),
    }
    (OUTPUT_DIR / "train_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved training config to {OUTPUT_DIR / 'train_config.json'}")

    # ----------------- Train & save -----------------------
    print("Starting training...")
    trainer.train()
    trainer.save_state()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
