"""
GLAR - Global-Local Attentive Relation Detection
=================================================

Predicts the relation expressed by a short text (here: a SimpleQuestions v2
question) using a BERT encoder followed by a global and a local attention
branch.

How it works
------------
1. The text is tokenized and entity markers are inserted:
       [S] subject [/S]   ...   [O] object [/O]
2. BERT encodes the marked text into one vector per token.
3. Two branches process those vectors in parallel:
       - Global branch: every token can attend to every other token.
       - Local branch : every token only attends to its close neighbours.
4. A fusion layer combines both branches with a learned gate.
5. The vectors at the subject and object markers are pooled together and
   passed to a small MLP that predicts the relation (one class per relation).

Usage
-----
    python glar_clean.py --data-dir ./SimpleQuestions_v2

The data directory must contain:
    annotated_fb_data_train.json
    annotated_fb_data_valid.json
    annotated_fb_data_test.json

Run `python glar_clean.py --help` to see all options.

Contents
--------
    1. Configuration
    2. Data loading and dataset
    3. Model components
    4. Full model
    5. Training utilities and evaluation
    6. Training loop
    7. Entry point
"""

import argparse
import json
import logging
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast

logger = logging.getLogger("GLAR")

# Special tokens that mark the subject and object inside the text.
SUBJ_START, SUBJ_END = "[S]", "[/S]"
OBJ_START, OBJ_END = "[O]", "[/O]"
MARKER_TOKENS = [SUBJ_START, SUBJ_END, OBJ_START, OBJ_END]

# Label used for relations that are not among the most frequent ones.
OTHER_LABEL = "__other__"


# ===========================================================================
# 1. Configuration
# ===========================================================================
@dataclass
class GLARConfig:
    """All hyper-parameters in one place."""

    # --- Data ---
    max_seq_len: int = 64          # SimpleQuestions are short; use 128 for longer sentences
    num_workers: int = 0           # 0 is the safest value on Windows

    # --- Encoder (BERT) ---
    bert_model_name: str = "bert-base-uncased"
    freeze_bert_layers: int = 8    # the first N BERT layers are not trained
    train_word_embeddings: bool = True   # needed so the new [S]/[O] tokens can learn

    # --- Global branch ---
    global_num_heads: int = 8
    global_num_layers: int = 2
    global_dropout: float = 0.1

    # --- Local branch ---
    local_window_size: int = 5     # neighbours looked at on each side of a token
    local_num_heads: int = 4
    local_num_layers: int = 2
    local_dropout: float = 0.1

    # --- Fusion and entity pooling ---
    fusion_num_heads: int = 8
    fusion_hidden_dim: int = 512   # size of the pooled subject/object vector
    fusion_dropout: float = 0.2
    entity_pool_span: int = 3      # tokens pooled from each marker: [S] word [/S]

    # --- Classifier ---
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.3

    # --- Training ---
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-5    # new layers (global, local, fusion, classifier)
    bert_lr: float = 1e-5          # BERT layers
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 100           # print the loss every N steps
    select_metric: str = "accuracy"  # metric used to pick the best epoch: "accuracy" or "f1"

    # --- Output ---
    output_dir: str = "outputs/"


# ===========================================================================
# 2. Data loading and dataset
# ===========================================================================
# A "record" is a plain dict with this structure:
#   {
#     "tokens":   ["what", "is", "the", "book", "e", "about"],
#     "h":        {"start": 0, "end": 1},     # subject span, end is exclusive
#     "t":        {"start": 5, "end": 6},     # object span,  end is exclusive
#     "relation": "book.written_work.subjects"
#   }
# ---------------------------------------------------------------------------
def build_tokenizer(model_name: str) -> BertTokenizerFast:
    """Load the BERT tokenizer and register the four entity marker tokens."""
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    return tokenizer


def insert_entity_markers(tokens: List[str], h: dict, t: dict) -> List[str]:
    """Return a copy of `tokens` with [S]/[/S] and [O]/[/O] inserted around the entities."""
    marked = []
    for i, tok in enumerate(tokens):
        if i == h["start"]:
            marked.append(SUBJ_START)
        if i == t["start"]:
            marked.append(OBJ_START)
        marked.append(tok)
        if i == h["end"] - 1:
            marked.append(SUBJ_END)
        if i == t["end"] - 1:
            marked.append(OBJ_END)
    return marked


class RelationDataset(Dataset):
    """Turns records into fixed-size BERT inputs (tokenized once, kept in memory)."""

    def __init__(
        self,
        records: List[dict],
        tokenizer: BertTokenizerFast,
        max_seq_len: int,
        label2id: Dict[str, int],
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = label2id

        # Look the marker ids up once (doing this per example is very slow).
        self.subj_id = tokenizer.convert_tokens_to_ids(SUBJ_START)
        self.obj_id = tokenizer.convert_tokens_to_ids(OBJ_START)
        if tokenizer.unk_token_id in (self.subj_id, self.obj_id):
            raise ValueError("Marker tokens are not registered. Use build_tokenizer().")

        self.examples = [self._encode(r) for r in records]

    def _encode(self, record: dict) -> dict:
        marked = insert_entity_markers(record["tokens"], record["h"], record["t"])
        enc = self.tokenizer(
            marked,
            is_split_into_words=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        ids = enc["input_ids"]

        # Position of each start marker. If a marker was cut off by truncation,
        # fall back to position 0 ([CLS]) so the model still gets a valid input.
        subj_pos = ids.index(self.subj_id) if self.subj_id in ids else 0
        obj_pos = ids.index(self.obj_id) if self.obj_id in ids else 0

        # Rare relations map to the "__other__" class when that class exists.
        relation = record["relation"]
        if relation in self.label2id:
            label = self.label2id[relation]
        elif OTHER_LABEL in self.label2id:
            label = self.label2id[OTHER_LABEL]
        else:
            raise KeyError(f"Unknown relation '{relation}' and no '{OTHER_LABEL}' class.")

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(enc["token_type_ids"], dtype=torch.long),
            "subj_pos": torch.tensor(subj_pos, dtype=torch.long),
            "obj_pos": torch.tensor(obj_pos, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def load_json_records(path: str) -> List[dict]:
    """Load records from a JSON file that already uses the record format above."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_sparql(sparql: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (subject_mid, relation) from a SimpleQuestions SPARQL query.

    The query has the form:  ... { <.../SUBJECT_MID> <.../RELATION> ?uri }
    """
    match = re.search(r"\{\s*<[^>]*/([^/>]+)>\s+<[^>]*/([^/>]+)>\s+\?", sparql)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_simplequestions_v2(folder: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Read the three SimpleQuestions v2 files and convert them to records.

    The subject only exists as a Freebase id (it has no surface form in the
    question), so the first word is used as the subject anchor and the last word
    as the object anchor. This gives the model fixed positions to pool from.
    """
    files = {
        "train": "annotated_fb_data_train.json",
        "dev": "annotated_fb_data_valid.json",
        "test": "annotated_fb_data_test.json",
    }

    splits = {}
    for split, filename in files.items():
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}. Check the --data-dir option.")
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        records, skipped = [], 0
        for item in questions:
            text = item.get("question", "").strip()
            _, relation = _parse_sparql(item["query"]["sparql"])
            tokens = text.split()
            if not tokens or not relation:
                skipped += 1
                continue
            records.append({
                "tokens": tokens,
                "h": {"start": 0, "end": 1},
                "t": {"start": len(tokens) - 1, "end": len(tokens)},
                "relation": relation,
            })

        logger.info(f"  {split}: {len(records):,} records loaded ({skipped} skipped)")
        splits[split] = records

    return splits["train"], splits["dev"], splits["test"]


def build_label_maps(
    train_records: List[dict], max_relations: Optional[int]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create the relation <-> id maps from the training records.

    If `max_relations` is set, only the most frequent relations get their own
    class and all others share the "__other__" class.
    """
    counts = Counter(r["relation"] for r in train_records)
    if max_relations is None:
        names = sorted(counts)
    else:
        names = sorted(rel for rel, _ in counts.most_common(max_relations)) + [OTHER_LABEL]

    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for name, i in label2id.items()}
    logger.info(f"  Label space: {len(label2id)} classes ({len(counts)} relations in train)")
    return label2id, id2label


def build_dataloaders(
    train_records: List[dict],
    dev_records: List[dict],
    test_records: List[dict],
    tokenizer: BertTokenizerFast,
    config: GLARConfig,
    label2id: Dict[str, int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train / dev / test DataLoaders."""
    pin_memory = torch.cuda.is_available()

    def make(records: List[dict], shuffle: bool) -> DataLoader:
        dataset = RelationDataset(records, tokenizer, config.max_seq_len, label2id)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )

    return make(train_records, True), make(dev_records, False), make(test_records, False)


# ===========================================================================
# 3. Model components
# ===========================================================================
class GlobalAttentionLayer(nn.Module):
    """One transformer block: full self-attention + feed-forward, each with residual and LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ff(x)))


class GlobalBranch(nn.Module):
    """Stack of global attention layers (captures long-range relations between tokens)."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [GlobalAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x  # (B, L, D)


class LocalWindowAttention(nn.Module):
    """
    Windowed self-attention block.

    Each token attends only to the tokens within +/- `window_size` positions,
    which focuses the layer on nearby words.
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        W = self.window_size
        window_len = 2 * W + 1

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, head_dim)

        # Pad the sequence by W on both sides, then slice a window of size 2W+1
        # for every position. unfold() does this without a Python loop.
        # Result shape: (B, L, H, head_dim, window_len)
        k_win = F.pad(k, (0, 0, 0, 0, W, W)).unfold(1, window_len, 1)
        v_win = F.pad(v, (0, 0, 0, 0, W, W)).unfold(1, window_len, 1)

        # Attention scores between each token and the tokens in its window.
        scores = torch.einsum("blhd,blhdw->blhw", q, k_win) * self.scale  # (B, L, H, window_len)

        # Block attention to padding tokens and to the artificial padding at the edges.
        if pad_mask is None:
            pad_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        edge_padded = F.pad(pad_mask, (W, W), value=True)                   # (B, L + 2W)
        win_mask = edge_padded.unfold(1, window_len, 1).unsqueeze(2)        # (B, L, 1, window_len)
        # A large negative number (not -inf) keeps fully masked rows finite, so no NaN can appear.
        scores = scores.masked_fill(win_mask, torch.finfo(scores.dtype).min)

        weights = self.drop(F.softmax(scores, dim=-1))
        out = torch.einsum("blhw,blhdw->blhd", weights, v_win).reshape(B, L, D)
        out = self.proj(out)

        x = self.norm1(x + self.drop(out))
        return self.norm2(x + self.drop(self.ff(x)))


class LocalBranch(nn.Module):
    """Stack of windowed attention layers (captures nearby word interactions)."""

    def __init__(self, d_model: int, num_heads: int, window_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [LocalWindowAttention(d_model, num_heads, window_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x  # (B, L, D)


class GLFusionLayer(nn.Module):
    """
    Combines the global and local representations.

      1. Global tokens look at the local tokens (cross-attention).
      2. Local tokens look at the global tokens (cross-attention).
      3. A learned gate (values between 0 and 1) decides, per feature, how much
         to take from each result.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_g2l = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_l2g = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm_g = nn.LayerNorm(d_model)
        self.norm_l = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        global_rep: torch.Tensor,
        local_rep: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g2l, _ = self.cross_g2l(global_rep, local_rep, local_rep, key_padding_mask=pad_mask)
        g2l = self.norm_g(global_rep + self.drop(g2l))

        l2g, _ = self.cross_l2g(local_rep, global_rep, global_rep, key_padding_mask=pad_mask)
        l2g = self.norm_l(local_rep + self.drop(l2g))

        gate = self.gate(torch.cat([g2l, l2g], dim=-1))  # (B, L, D)
        return self.norm_out(gate * g2l + (1 - gate) * l2g)


class EntityPooler(nn.Module):
    """
    Builds one vector for the (subject, object) pair.

    For each entity, the vectors of `span` tokens starting at its start marker
    are averaged (default: [S], the word, [/S]). Padding tokens are ignored.
    The two entity vectors are concatenated and projected to `hidden_dim`.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float, span: int = 3):
        super().__init__()
        self.span = span
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def _pool_at(self, seq: torch.Tensor, attention_mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Average `span` token vectors starting at `pos` for every sample in the batch."""
        B, L, D = seq.shape
        idx = pos.unsqueeze(1) + torch.arange(self.span, device=seq.device)  # (B, span)
        in_range = idx < L
        idx = idx.clamp(max=L - 1)

        valid = in_range & attention_mask.gather(1, idx).bool()              # (B, span)
        picked = seq.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))          # (B, span, D)

        weights = valid.unsqueeze(-1).to(seq.dtype)
        return (picked * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)  # (B, D)

    def forward(
        self,
        seq: torch.Tensor,
        attention_mask: torch.Tensor,
        subj_pos: torch.Tensor,
        obj_pos: torch.Tensor,
    ) -> torch.Tensor:
        subj = self._pool_at(seq, attention_mask, subj_pos)
        obj = self._pool_at(seq, attention_mask, obj_pos)
        return self.proj(torch.cat([subj, obj], dim=-1))  # (B, hidden_dim)


# ===========================================================================
# 4. Full model
# ===========================================================================
class GLARModel(nn.Module):
    """
    BERT -> global branch + local branch -> fusion -> entity pooling -> MLP -> logits
    """

    def __init__(self, config: GLARConfig, num_labels: int, vocab_size: int):
        super().__init__()

        # --- Encoder ---
        # The pooler layer of BERT is not used here, so it is not loaded.
        self.bert = BertModel.from_pretrained(config.bert_model_name, add_pooling_layer=False)
        # Make room for the [S], [/S], [O], [/O] tokens BEFORE freezing anything.
        self.bert.resize_token_embeddings(vocab_size)
        d_model = self.bert.config.hidden_size
        self._freeze_bert(config.freeze_bert_layers, config.train_word_embeddings)

        # --- Branches, fusion, pooling, classifier ---
        self.global_branch = GlobalBranch(
            d_model, config.global_num_heads, config.global_num_layers, config.global_dropout
        )
        self.local_branch = LocalBranch(
            d_model, config.local_num_heads, config.local_window_size,
            config.local_num_layers, config.local_dropout,
        )
        self.fusion = GLFusionLayer(d_model, config.fusion_num_heads, config.fusion_dropout)
        self.entity_pooler = EntityPooler(
            d_model, config.fusion_hidden_dim, config.fusion_dropout, config.entity_pool_span
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, num_labels),
        )

    def _freeze_bert(self, n_layers: int, train_word_embeddings: bool) -> None:
        """Freeze the embeddings and the first `n_layers` BERT layers."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        if train_word_embeddings:
            # Keep word embeddings trainable so the new marker tokens can be learned.
            self.bert.embeddings.word_embeddings.weight.requires_grad = True
        for layer in self.bert.encoder.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        subj_pos: torch.Tensor,
        obj_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return relation logits of shape (batch, num_labels)."""
        seq = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state                                   # (B, L, D)

        pad_mask = attention_mask == 0                        # True where the token is padding

        global_out = self.global_branch(seq, pad_mask)
        local_out = self.local_branch(seq, pad_mask)
        fused = self.fusion(global_out, local_out, pad_mask)

        pair = self.entity_pooler(fused, attention_mask, subj_pos, obj_pos)
        return self.classifier(pair)


# ===========================================================================
# 5. Training utilities and evaluation
# ===========================================================================
def set_seed(seed: int) -> None:
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer_and_scheduler(model: GLARModel, config: GLARConfig, num_steps: int):
    """
    AdamW with two learning rates (BERT vs. new layers). Biases and LayerNorm
    weights (1-D tensors) are excluded from weight decay.

    The learning rate rises linearly during warm-up, then decays linearly to zero.
    """
    groups = []
    for is_bert, lr in ((True, config.bert_lr), (False, config.learning_rate)):
        for decay in (True, False):
            params = [
                p for name, p in model.named_parameters()
                if p.requires_grad
                and name.startswith("bert.") == is_bert
                and (p.ndim >= 2) == decay
            ]
            if params:
                groups.append({
                    "params": params,
                    "lr": lr,
                    "weight_decay": config.weight_decay if decay else 0.0,
                })
    optimizer = AdamW(groups)

    warmup_steps = int(num_steps * config.warmup_ratio)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (num_steps - step) / max(1, num_steps - warmup_steps))

    return optimizer, LambdaLR(optimizer, lr_factor)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: GLARModel,
    loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    split: str = "dev",
    full_report: bool = False,
) -> Dict[str, float]:
    """
    Compute accuracy and macro precision / recall / F1.

    Macro scores are averaged over the classes that appear in the gold labels,
    so classes missing from a split do not lower the score.
    """
    model.eval()
    preds, golds = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
            batch["subj_pos"], batch["obj_pos"],
        )
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        golds.extend(batch["label"].cpu().tolist())

    labels = sorted(set(golds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        golds, preds, labels=labels, average="macro", zero_division=0
    )
    accuracy = float(np.mean(np.array(preds) == np.array(golds)))

    logger.info(
        f"[{split}] Acc={accuracy:.4f}  Precision={precision:.4f}  "
        f"Recall={recall:.4f}  F1={f1:.4f}"
    )
    if full_report:
        names = [id2label.get(i, str(i)) for i in labels]
        logger.info("\n" + classification_report(
            golds, preds, labels=labels, target_names=names, zero_division=0
        ))
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ===========================================================================
# 6. Training loop
# ===========================================================================
def train(
    model: GLARModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    id2label: Dict[int, str],
    config: GLARConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train the model, keep the best checkpoint according to the dev set,
    then evaluate that checkpoint on the test set.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    ckpt_path = os.path.join(config.output_dir, "best_model.pt")

    num_steps = len(train_loader) * config.num_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, num_steps)
    criterion = nn.CrossEntropyLoss()

    best_score, best_epoch = -1.0, 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = _to_device(batch, device)

            optimizer.zero_grad()
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                batch["subj_pos"], batch["obj_pos"],
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % config.log_every == 0:
                logger.info(
                    f"  Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"Loss={total_loss / step:.4f}"
                )

        logger.info(f"Epoch {epoch} done | Avg loss={total_loss / len(train_loader):.4f}")

        # Validate after every epoch and keep the best model.
        metrics = evaluate(model, dev_loader, device, id2label, split="dev")
        score = metrics["accuracy" if config.select_metric == "accuracy" else "f1"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  New best {config.select_metric}={score:.4f} - saved to {ckpt_path}")

    logger.info(f"Training complete. Best dev {config.select_metric}={best_score:.4f} (epoch {best_epoch})")

    # Final test evaluation with the best checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return evaluate(model, test_loader, device, id2label, split="test", full_report=True)


# ===========================================================================
# 7. Entry point
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GLAR on SimpleQuestions v2.")
    parser.add_argument("--data-dir", default="./SimpleQuestions_v2",
                        help="folder with the annotated_fb_data_*.json files")
    parser.add_argument("--max-relations", type=int, default=200,
                        help="number of most frequent relations to keep (0 = keep all)")
    parser.add_argument("--bert-model", default="bert-base-uncased",
                        help="Hugging Face model name or local folder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Start from the defaults and apply any command-line overrides.
    config = GLARConfig(bert_model_name=args.bert_model)
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    max_relations = args.max_relations or None

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Data ---
    tokenizer = build_tokenizer(config.bert_model_name)
    logger.info(f"Loading SimpleQuestions v2 from: {args.data_dir}")
    train_rec, dev_rec, test_rec = load_simplequestions_v2(args.data_dir)
    label2id, id2label = build_label_maps(train_rec, max_relations)
    train_loader, dev_loader, test_loader = build_dataloaders(
        train_rec, dev_rec, test_rec, tokenizer, config, label2id
    )

    # Save what is needed to use the model later.
    tokenizer.save_pretrained(config.output_dir)
    with open(os.path.join(config.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    # --- Model ---
    model = GLARModel(config, num_labels=len(label2id), vocab_size=len(tokenizer)).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters - total: {total:,}  trainable: {trainable:,}")

    # --- Train and test ---
    train(model, train_loader, dev_loader, test_loader, id2label, config, device)


if __name__ == "__main__":
    main()
