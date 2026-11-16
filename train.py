"""
train.py – Training script for the SBR-KV vs GQA research repository.

Usage:
    python train.py --model gqa --epochs 3 --batch_size 16
    python train.py --model sbr  --dataset tinystories --max_steps 500
"""

import argparse
import csv
import os
import sys
import time
import warnings
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GQA or SBR-KV transformer on TinyStories / Wikitext."
    )
    parser.add_argument(
        "--model", choices=["gqa", "sbr"], default="gqa",
        help="Which model architecture to train (default: gqa)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size (default: 16)."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate for AdamW (default: 3e-4)."
    )
    parser.add_argument(
        "--seq_len", type=int, default=128,
        help="Fixed sequence length to pack tokens into (default: 128)."
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)."
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Hard cap on total training steps (default: 500)."
    )
    parser.add_argument(
        "--dataset", choices=["tinystories", "wikitext"], default="tinystories",
        help="Dataset to use (default: tinystories)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for checkpoints and logs (default: ./outputs)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        warnings.warn(
            "CUDA is not available. Training on CPU – this will be slow.",
            UserWarning,
        )
        print("[device] Falling back to CPU.")
    return device


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class PackedTokenDataset(Dataset):
    """Packs a flat list of token IDs into fixed-length chunks for causal LM."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Drop the last incomplete chunk
        n_full = len(token_ids) // (seq_len + 1)
        self.data = torch.tensor(
            token_ids[: n_full * (seq_len + 1)], dtype=torch.long
        ).view(n_full, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]     # target tokens (shifted by 1)
        return x, y


class SimpleStoryTokenizer:
    """Offline vocabulary / tokenizer with explicit sentence boundary tokens."""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.boundary_ids = [10, 11, 12]  # '.', '!', '?'
        self.word2id = {"<eos>": 0, "<pad>": 1, ".": 10, "!": 11, "?": 12}
        self.id2word = {0: "<eos>", 1: "<pad>", 10: ".", 11: "!", 12: "?"}

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.replace(".", " . ").replace("!", " ! ").replace("?", " ? ").split():
            w = word.strip().lower()
            if not w:
                continue
            if w in self.word2id:
                tokens.append(self.word2id[w])
            else:
                if len(self.word2id) < self.vocab_size - 100:
                    idx = len(self.word2id) + 15
                    self.word2id[w] = idx
                    self.id2word[idx] = w
                    tokens.append(idx)
                else:
                    h = abs(hash(w)) % (self.vocab_size - 100) + 50
                    tokens.append(h)
        return tokens


def generate_synthetic_tinystories(n_samples: int = 5000) -> list[str]:
    """Generate realistic synthetic TinyStories passages for fast offline training/testing."""
    import random
    subjects = ["Once upon a time, Lily", "A little dog named Max", "Tim", "A happy little bird", "Sara", "A friendly bear", "A clever rabbit"]
    verbs = ["loved to play with", "found a secret in", "walked towards", "shared some food with", "saw a shiny", "discovered"]
    objects = ["the big blue ball", "a tall green tree", "a magic box", "the warm sunny garden", "a tiny flower", "a golden key"]
    adverbs = ["happily", "carefully", "suddenly", "very quickly", "with great joy", "gently"]
    conclusions = ["They smiled together.", "It was a wonderful day!", "Can you believe what happened?", "Everyone was so proud.", "They lived happily ever after."]

    rng = random.Random(42)
    stories = []
    for _ in range(n_samples):
        s1 = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(adverbs)}."
        s2 = f"Then, {rng.choice(subjects).lower()} {rng.choice(verbs)} {rng.choice(objects)}!"
        s3 = f"Did they find {rng.choice(objects)}? Yes, they did."
        s4 = rng.choice(conclusions)
        stories.append(f"{s1} {s2} {s3} {s4}")
    return stories


def load_and_tokenize(dataset_name: str, tokenizer, seq_len: int):
    """
    Load a dataset, tokenize, and return (train_dataset, val_dataset).

    Uses the first 5 000 examples for training and the first 500 for validation.
    Falls back gracefully to offline synthetic TinyStories if network is unavailable.
    """
    train_ids: list[int] = []
    val_ids: list[int] = []
    text_key = "text"

    try:
        from datasets import load_dataset
        print(f"[data] Attempting to load dataset '{dataset_name}' from HuggingFace Hub …")
        if dataset_name == "tinystories":
            hf_name = "roneneldan/TinyStories"
            train_raw = load_dataset(hf_name, split="train[:5000]")
            val_raw = load_dataset(hf_name, split="validation[:500]")
        else:
            hf_name = "wikitext"
            train_raw = load_dataset(hf_name, "wikitext-2-raw-v1", split="train[:5000]")
            val_raw = load_dataset(hf_name, "wikitext-2-raw-v1", split="validation[:500]")

        def tokenize_split(raw_split):
            all_ids: list[int] = []
            for example in raw_split:
                text = example[text_key]
                if not text.strip():
                    continue
                ids = tokenizer.encode(text)
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)
            return all_ids

        train_ids = tokenize_split(train_raw)
        val_ids = tokenize_split(val_raw)
    except Exception as e:
        print(f"[data] Offline fallback: could not fetch remote dataset ({e}).")
        print("[data] Generating synthetic TinyStories corpus locally (5,000 train / 500 val passages) …")
        train_texts = generate_synthetic_tinystories(5000)
        val_texts = generate_synthetic_tinystories(500)

        for text in train_texts:
            train_ids.extend(tokenizer.encode(text))
            train_ids.append(tokenizer.eos_token_id)

        for text in val_texts:
            val_ids.extend(tokenizer.encode(text))
            val_ids.append(tokenizer.eos_token_id)

    print(f"[data] Train tokens: {len(train_ids):,}")
    print(f"[data] Val tokens:   {len(val_ids):,}")

    train_ds = PackedTokenDataset(train_ids, seq_len)
    val_ds = PackedTokenDataset(val_ids, seq_len)

    print(f"[data] Train sequences: {len(train_ds):,}  |  Val sequences: {len(val_ds):,}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 10 % of peak LR."""
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int = 25) -> float:
    """Return mean cross-entropy loss over the validation set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, T, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # -- Reproducibility -------------------------------------------------
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()

    # -- Output directory ------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Tokenizer -------------------------------------------------------
    boundary_ids = [10, 11, 12]  # '.', '!', '?'
    try:
        from transformers import GPT2TokenizerFast
        # Test if weights/vocab can be fetched quickly
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=False)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        period_toks = tokenizer.encode(".")
        excl_toks = tokenizer.encode("!")
        quest_toks = tokenizer.encode("?")
        if period_toks: boundary_ids = [period_toks[0]]
        if excl_toks: boundary_ids.append(excl_toks[0])
        if quest_toks: boundary_ids.append(quest_toks[0])
        print(f"[tokenizer] Loaded GPT-2 tokenizer (vocab_size={vocab_size:,})")
    except Exception as e:
        print(f"[tokenizer] Offline mode: using SimpleStoryTokenizer ({e})")
        tokenizer = SimpleStoryTokenizer(vocab_size=10000)
        vocab_size = tokenizer.vocab_size
        boundary_ids = tokenizer.boundary_ids

    # -- Data ------------------------------------------------------------
    train_ds, val_ds = load_and_tokenize(args.dataset, tokenizer, args.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda"
    )

    # -- Model -----------------------------------------------------------
    # Import from model.py in the same directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from model import GQATransformer, SBRTransformer, count_params  # noqa: E402

    model_cfg = dict(
        vocab_size=vocab_size,
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        ffn_mult=4,
    )

    if args.model == "gqa":
        model = GQATransformer(**model_cfg)
    else:
        model = SBRTransformer(**model_cfg, boundary_ids=boundary_ids)

    model = model.to(device)
    n_params = count_params(model)
    print(f"[model] {args.model.upper()} | Parameters: {n_params:,}")

    # -- Optimizer & schedule -------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = min(args.max_steps, args.epochs * len(train_loader))
    warmup_steps = max(1, total_steps // 20)          # 5 % warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, warmup_steps, total_steps),
    )

    criterion = nn.CrossEntropyLoss()

    # -- CSV log ---------------------------------------------------------
    csv_path = out_dir / f"train_log_{args.model}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "tokens_per_sec"])

    # -- Training --------------------------------------------------------
    model.train()
    global_step = 0
    epoch_num = 0
    running_loss = 0.0
    running_tokens = 0
    t0 = time.perf_counter()
    last_val_loss = float("nan")
    all_tokens_per_sec: list[float] = []

    print(f"\n[train] Starting training for up to {total_steps} steps …\n")

    outer_break = False
    for epoch in range(args.epochs):
        if outer_break:
            break
        epoch_num = epoch + 1

        for x, y in train_loader:
            if global_step >= total_steps:
                outer_break = True
                break

            x, y = x.to(device), y.to(device)
            batch_tokens = x.numel()

            step_t0 = time.perf_counter()

            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step_t1 = time.perf_counter()
            step_secs = step_t1 - step_t0
            tokens_per_sec = batch_tokens / max(step_secs, 1e-9)

            global_step += 1
            running_loss += loss.item()
            running_tokens += batch_tokens
            all_tokens_per_sec.append(tokens_per_sec)

            # -- Logging every 50 steps --
            if global_step % 50 == 0 or global_step == 1:
                avg_loss = running_loss / (global_step if global_step < 50 else 50)
                # Reset running window after each print
                if global_step >= 50:
                    running_loss = 0.0

                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[step {global_step:>5}]  loss={avg_loss:.4f}  "
                    f"val_loss={last_val_loss:.4f}  "
                    f"tok/s={tokens_per_sec:,.0f}  lr={lr_now:.2e}"
                )
                csv_writer.writerow([global_step, f"{avg_loss:.6f}",
                                     f"{last_val_loss:.6f}", f"{tokens_per_sec:.1f}"])
                csv_file.flush()

            # -- Validation every 100 steps --
            if global_step % 100 == 0:
                last_val_loss = evaluate(model, val_loader, device)
                print(
                    f"  >> [val  {global_step:>5}]  val_loss={last_val_loss:.4f}"
                )

    csv_file.close()
    print(f"\n[train] Training finished after {global_step} steps.")

    # Final validation pass
    print("[val] Running final validation …")
    final_val_loss = evaluate(model, val_loader, device)
    print(f"[val] Final val loss: {final_val_loss:.4f}")

    # -- Save checkpoint ------------------------------------------------
    ckpt_path = out_dir / f"{args.model}_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": global_step,
            "val_loss": final_val_loss,
        },
        ckpt_path,
    )
    print(f"[ckpt] Saved checkpoint → {ckpt_path}")

    # -- Summary table --------------------------------------------------
    avg_tps = sum(all_tokens_per_sec) / max(1, len(all_tokens_per_sec))

    col_w = 22
    sep = "+" + "-" * col_w + "+" + "-" * col_w + "+"
    print("\n" + sep)
    print(f"| {'Metric':<{col_w - 2}} | {'Value':<{col_w - 2}} |")
    print(sep)
    print(f"| {'Model':<{col_w - 2}} | {args.model.upper():<{col_w - 2}} |")
    print(f"| {'Parameters':<{col_w - 2}} | {f'{n_params:,}':<{col_w - 2}} |")
    print(f"| {'Steps trained':<{col_w - 2}} | {global_step:<{col_w - 2}} |")
    print(f"| {'Final val loss':<{col_w - 2}} | {f'{final_val_loss:.4f}':<{col_w - 2}} |")
    print(f"| {'Avg tok/s':<{col_w - 2}} | {f'{avg_tps:,.1f}':<{col_w - 2}} |")
    print(sep + "\n")

    return final_val_loss, avg_tps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
