#!/usr/bin/env python3
"""
Edit tagger training script (Ziętkiewicz, 2024 style) for ASR post-editing.

Usage (two GPUs on SLURM node):
    torchrun --nproc_per_node=2 edit_tagger_train.py

This version trains on a single CSV file:
    emb_scores_clean_filtered.csv

Columns used:
    - transcript_norm : noisy/ASR-like text
    - sentence_norm   : clean reference text
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path("emb_scores_clean_strict_train.csv")
OUTPUT_DIR = Path("edit_tagger_model")
MODEL_NAME = "xlm-roberta-base"
INSERT_PLACEHOLDER = "<ins>"

VAL_FRACTION = 0.1
MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 50
RANDOM_SEED = 42
NUM_WORKERS = 4
TAG_LOSS_WEIGHT = 1.0
TOKEN_LOSS_WEIGHT = 1.0

TAGS = ["KEEP", "DELETE", "REPLACE", "INSERT", "PAD"]
TAG2ID = {tag: idx for idx, tag in enumerate(TAGS)}
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()}
TAG_PAD_ID = TAG2ID["PAD"]
NEEDS_EDIT_TOKEN = {TAG2ID["REPLACE"], TAG2ID["INSERT"]}
EDIT_IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def log(message: str, rank: int, log_file: Path) -> None:
    if is_main_process(rank):
        print(message, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def levenshtein_alignment(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Returns list of operations: (op, hyp_word, ref_word)
    op in {'equal','sub','del','ins'}
    """
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # insertion (ref)
                dp[i][j - 1] + 1,      # deletion (hyp)
                dp[i - 1][j - 1] + cost,  # replace/keep
            )
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref_words[i - 1] == hyp_words[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            ops.append(("equal", hyp_words[j - 1], ref_words[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", hyp_words[j - 1], ref_words[i - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("del", hyp_words[j - 1], None))
            j -= 1
        else:
            ops.append(("ins", None, ref_words[i - 1]))
            i -= 1
    ops.reverse()
    return ops


def build_edit_sequence(asr_text: str, ref_text: str) -> Tuple[List[str], List[str], List[Optional[str]]]:
    hyp_words = asr_text.strip().split()
    ref_words = ref_text.strip().split()
    if not hyp_words:
        hyp_words = ["<blank>"]
    ops = levenshtein_alignment(ref_words, hyp_words)
    tokens: List[str] = []
    tags: List[str] = []
    edit_words: List[Optional[str]] = []
    for op, hyp_word, ref_word in ops:
        if op == "equal":
            tokens.append(hyp_word or "")
            tags.append("KEEP")
            edit_words.append(None)
        elif op == "sub":
            tokens.append(hyp_word or "")
            tags.append("REPLACE")
            edit_words.append(ref_word or "")
        elif op == "del":
            tokens.append(hyp_word or "")
            tags.append("DELETE")
            edit_words.append(None)
        elif op == "ins":
            tokens.append(INSERT_PLACEHOLDER)
            tags.append("INSERT")
            edit_words.append(ref_word or "")
    return tokens, tags, edit_words


@dataclass
class EditExample:
    tokens: List[str]
    tags: List[str]
    edit_words: List[Optional[str]]


class EditTaggerDataset(Dataset):
    def __init__(self, examples: List[EditExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        enc = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = enc.word_ids(batch_index=0)
        tag_ids: List[int] = []
        edit_token_ids: List[int] = []
        for wi in word_ids:
            if wi is None:
                tag_ids.append(TAG_PAD_ID)
                edit_token_ids.append(EDIT_IGNORE_INDEX)
            else:
                tag = example.tags[wi]
                tag_id = TAG2ID.get(tag, TAG_PAD_ID)
                tag_ids.append(tag_id)
                if tag_id in NEEDS_EDIT_TOKEN:
                    edit_word = example.edit_words[wi] or ""
                    token_ids = self.tokenizer.encode(edit_word, add_special_tokens=False)
                    edit_token_ids.append(
                        token_ids[0] if token_ids else self.tokenizer.unk_token_id
                    )
                else:
                    edit_token_ids.append(EDIT_IGNORE_INDEX)

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        tag_tensor = torch.tensor(tag_ids, dtype=torch.long)
        edit_tensor = torch.tensor(edit_token_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tag_ids": tag_tensor,
            "edit_token_ids": edit_tensor,
        }


class EditTaggerModel(nn.Module):
    def __init__(self, base_model_name: str, num_tags: int, vocab_size: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.tag_classifier = nn.Linear(hidden_size, num_tags)
        self.token_classifier = nn.Linear(hidden_size, vocab_size)
        self.tag_loss_fct = nn.CrossEntropyLoss(ignore_index=TAG_PAD_ID)
        self.token_loss_fct = nn.CrossEntropyLoss(ignore_index=EDIT_IGNORE_INDEX)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tag_ids: Optional[torch.Tensor] = None,
        edit_token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tag_logits = self.tag_classifier(sequence_output)
        token_logits = self.token_classifier(sequence_output)
        loss = None
        if tag_ids is not None and edit_token_ids is not None:
            tag_loss = self.tag_loss_fct(
                tag_logits.view(-1, tag_logits.size(-1)),
                tag_ids.view(-1),
            )
            token_loss = self.token_loss_fct(
                token_logits.view(-1, token_logits.size(-1)),
                edit_token_ids.view(-1),
            )
            loss = TAG_LOSS_WEIGHT * tag_loss + TOKEN_LOSS_WEIGHT * token_loss
        return {"loss": loss, "tag_logits": tag_logits, "token_logits": token_logits}


def gather_list(data: List[int]) -> List[int]:
    if not dist.is_initialized():
        return data
    world_size = dist.get_world_size()
    gather_objs: List[Optional[List[int]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gather_objs, data)
    merged: List[int] = []
    for obj in gather_objs:
        if obj:
            merged.extend(obj)
    return merged


def evaluate(model, data_loader, device, rank: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in data_loader:
            total_batches += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            total_loss += loss.item()
            tag_logits = outputs["tag_logits"]
            tag_preds = torch.argmax(tag_logits, dim=-1)
            mask = batch["tag_ids"] != TAG_PAD_ID
            preds.extend(tag_preds[mask].detach().cpu().tolist())
            labels.extend(batch["tag_ids"][mask].detach().cpu().tolist())

    if dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, total_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        total_batches = int(loss_tensor[1].item())

    preds = gather_list(preds)
    labels = gather_list(labels)
    avg_loss = total_loss / max(total_batches, 1)

    if len(labels) == 0:
        return {"loss": avg_loss, "accuracy": 0.0, "f1": 0.0}

    correct = sum(int(p == l) for p, l in zip(preds, labels))
    accuracy = correct / len(labels)
    valid_labels = [TAG2ID["KEEP"], TAG2ID["DELETE"], TAG2ID["REPLACE"], TAG2ID["INSERT"]]
    f1 = f1_score(labels, preds, labels=valid_labels, average="macro", zero_division=0)
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def load_dataset(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")
    df = pd.read_csv(data_file)
    required_cols = {"sentence_norm", "transcript_norm"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"File {data_file} must contain columns {required_cols}, "
            f"but has {set(df.columns)}"
        )
    return df


def build_examples(df: pd.DataFrame) -> List[EditExample]:
    examples: List[EditExample] = []
    for asr_text, ref_text in zip(df["transcript_norm"], df["sentence_norm"]):
        tokens, tags, edit_words = build_edit_sequence(str(asr_text), str(ref_text))
        examples.append(EditExample(tokens, tags, edit_words))
    return examples


def sync_is_finite(loss: torch.Tensor, device: torch.device) -> bool:
    """
    DDP-sicherer NaN/Inf-Check:
    - lokal is_finite berechnen
    - via all_reduce (MIN) über alle Ranks synchronisieren
    """
    is_finite_local = torch.isfinite(loss.detach())
    if not dist.is_initialized():
        return bool(is_finite_local)

    flag = torch.tensor(1.0 if is_finite_local else 0.0, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() == 1.0)


def main():
    rank, world_size, local_rank = init_distributed()
    set_seed(RANDOM_SEED + rank)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "training.log"

    if is_main_process(rank):
        with log_file.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_acc,val_f1\n")
        config = {
            "data_file": str(DATA_FILE),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "tag_loss_weight": TAG_LOSS_WEIGHT,
            "token_loss_weight": TOKEN_LOSS_WEIGHT,
            "val_fraction": VAL_FRACTION,
        }
        with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        with (OUTPUT_DIR / "labels.json").open("w", encoding="utf-8") as f:
            json.dump({"label2id": TAG2ID, "id2label": ID2TAG}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Dataset laden & splitten
    full_df = load_dataset(DATA_FILE)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_examples = build_examples(train_df)
    valid_examples = build_examples(valid_df)
    train_dataset = EditTaggerDataset(train_examples, tokenizer, MAX_LENGTH)
    valid_dataset = EditTaggerDataset(valid_examples, tokenizer, MAX_LENGTH)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = EditTaggerModel(MODEL_NAME, len(TAGS), tokenizer.vocab_size).to(device)
    ddp_model = DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )

    optimizer = AdamW(ddp_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader) * NUM_EPOCHS, 1)
    warmup_steps = min(WARMUP_STEPS, max(total_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_state_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0
        effective_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = ddp_model(**batch)
            loss = outputs["loss"]

            # DDP-sicherer NaN/Inf-Guard
            if not sync_is_finite(loss, device):
                if is_main_process(rank):
                    log(
                        f"Epoch {epoch} Step {step}: non-finite loss, skipping batch.",
                        rank,
                        log_file,
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            effective_steps += 1

            if step % LOGGING_STEPS == 0 and is_main_process(rank):
                avg = total_loss / max(effective_steps, 1)
                log(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} Loss {avg:.4f}",
                    rank,
                    log_file,
                )

        train_loss = total_loss / max(effective_steps, 1)
        val_metrics = evaluate(ddp_model, valid_loader, device, rank)

        log(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}",
            rank,
            log_file,
        )

        if val_metrics["f1"] > best_f1 and is_main_process(rank):
            best_f1 = val_metrics["f1"]
            torch.save(ddp_model.module.state_dict(), best_state_path)
            tokenizer.save_pretrained(OUTPUT_DIR)
            log(f"Saved new best model with F1={best_f1:.4f}", rank, log_file)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
