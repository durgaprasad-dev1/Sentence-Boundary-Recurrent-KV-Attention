"""Train and evaluate compact probes on synthetic associative recall."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


D_MODEL = 128
N_LAYERS = 2
TRAIN_STEPS = 300
TRAIN_CONTEXT_LENGTH = 128
BATCH_SIZE = 32
N_PAIRS = 5
N_VALUES = 40
VOCAB_SIZE = 128
PAIR_END_TOKEN = 1
FILLER_LOW = 82

VARIANTS = (
    "GQA Baseline",
    "SBR-KV with Learned EMA Gate",
    "SBR-KV with Unweighted Mean",
)


def build_recall_example(
    context_length: int, rng: random.Random
) -> Tuple[List[int], int]:
    """Create five non-overlapping key/value records and a query key."""
    if context_length < 3 * N_PAIRS:
        raise ValueError(f"context_length must be at least {3 * N_PAIRS}")

    tokens = [FILLER_LOW] * context_length

    starts = list(range(context_length - 2))
    rng.shuffle(starts)
    selected_starts: List[int] = []
    for candidate in starts:
        if all(abs(candidate - start) >= 3 for start in selected_starts):
            selected_starts.append(candidate)
            if len(selected_starts) == N_PAIRS:
                break

    keys = rng.sample(range(2, 2 + N_VALUES), N_PAIRS)
    values = rng.sample(range(2 + N_VALUES, 2 + 2 * N_VALUES), N_PAIRS)
    pairs = list(zip(keys, values))
    for start, (key, value) in zip(sorted(selected_starts), pairs):
        tokens[start : start + 3] = [key, value, PAIR_END_TOKEN]

    query_key, answer = rng.choice(pairs)
    return tokens + [query_key], answer - (2 + N_VALUES)


def make_batch(
    batch_size: int, context_length: int, rng: random.Random, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = [
        build_recall_example(context_length, rng) for _ in range(batch_size)
    ]
    inputs = torch.tensor([example[0] for example in examples], device=device)
    targets = torch.tensor([example[1] for example in examples], device=device)
    return inputs, targets


def sinusoidal_positions(length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, D_MODEL, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / D_MODEL)
    )
    angles = positions * frequencies
    encoding = torch.zeros(length, D_MODEL, device=device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding.unsqueeze(0)


class CausalAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = nn.MultiheadAttention(
            D_MODEL, num_heads=4, batch_first=True
        )
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.feed_forward(self.norm2(x))


class CompressedMemoryBlock(nn.Module):
    def __init__(self, use_ema_gate: bool) -> None:
        super().__init__()
        self.use_ema_gate = use_ema_gate
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.pair_projection = nn.Linear(2 * D_MODEL, D_MODEL)
        if use_ema_gate:
            self.memory = nn.GRU(D_MODEL, D_MODEL, batch_first=True)
        else:
            self.memory_projection = nn.Linear(D_MODEL, D_MODEL)
        self.state_projection = nn.Linear(D_MODEL, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )

    def forward(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boundary_positions = (input_ids == PAIR_END_TOKEN).nonzero(as_tuple=False)
        boundary_positions = boundary_positions[:, 1].view(x.size(0), N_PAIRS)
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        keys = x[batch_indices, boundary_positions - 2]
        values = x[batch_indices, boundary_positions - 1]
        pair_states = torch.tanh(
            self.pair_projection(torch.cat((keys, values), dim=-1))
        )
        if self.use_ema_gate:
            memory, _ = self.memory(self.norm1(pair_states))
            state = memory[:, -1]
        else:
            state = self.memory_projection(pair_states.mean(dim=1))
        x = x.clone()
        x[:, -1] = x[:, -1] + self.state_projection(state)
        x = x + self.feed_forward(self.norm2(x))
        return x, state


class RecallProbe(nn.Module):
    """Two-layer, 128-dimensional probe with the requested memory mechanism."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        if variant == VARIANTS[0]:
            self.layers = nn.ModuleList(
                CausalAttentionBlock() for _ in range(N_LAYERS)
            )
        else:
            self.layers = nn.ModuleList(
                CompressedMemoryBlock(use_ema_gate=variant == VARIANTS[1])
                for _ in range(N_LAYERS)
            )
        self.output_norm = nn.LayerNorm(D_MODEL)
        if variant == VARIANTS[0]:
            self.classifier = nn.Linear(D_MODEL, N_VALUES)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(2 * D_MODEL, D_MODEL),
                nn.GELU(),
                nn.Linear(D_MODEL, N_VALUES),
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * (D_MODEL**0.5)
        x = x + sinusoidal_positions(input_ids.size(1), input_ids.device)
        if self.variant == VARIANTS[0]:
            causal_mask = torch.triu(
                torch.ones(
                    input_ids.size(1),
                    input_ids.size(1),
                    device=input_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, causal_mask)
            return self.classifier(self.output_norm(x[:, -1]))
        else:
            for layer in self.layers:
                x, state = layer(x, input_ids)
            return self.classifier(
                torch.cat((self.output_norm(x[:, -1]), self.output_norm(state)), dim=-1)
            )


def train_probe(
    model: RecallProbe,
    device: torch.device,
    rng: random.Random,
    steps: int = TRAIN_STEPS,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        inputs, targets = make_batch(
            BATCH_SIZE, TRAIN_CONTEXT_LENGTH, rng, device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step in (1, 100, 200, steps):
            print(f"  step {step:>3}/{steps}: loss={loss.item():.4f}")


@torch.inference_mode()
def evaluate_probe(
    model: RecallProbe,
    context_lengths: List[int],
    n_trials: int,
    rng: random.Random,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    results: Dict[str, float] = {}
    eval_batch_size = 32
    for context_length in context_lengths:
        correct = 0
        remaining = n_trials
        while remaining:
            batch_size = min(eval_batch_size, remaining)
            inputs, targets = make_batch(batch_size, context_length, rng, device)
            correct += (model(inputs).argmax(dim=-1) == targets).sum().item()
            remaining -= batch_size
        results[str(context_length)] = correct / n_trials
        print(f"  context={context_length:>4}: accuracy={correct / n_trials:.4f}")
    return results


def save_results(results: dict, output_dir: Path) -> None:
    path = output_dir / "eval_results.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    print(f"[INFO] Results saved -> {path}")


def save_plot(
    accuracies: Dict[str, Dict[str, float]],
    context_lengths: List[int],
    output_dir: Path,
) -> None:
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    markers = ("o", "s", "^")
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, color, marker in zip(VARIANTS, colors, markers):
        values = [accuracies[variant][str(length)] for length in context_lengths]
        ax.plot(
            context_lengths,
            values,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=variant,
        )
    ax.set_title("Associative Recall Accuracy after Probe Training")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(context_lengths)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / "needle_recall_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lightweight probes for synthetic associative recall."
    )
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument(
        "--context_lengths", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "assets")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n_trials must be positive")
    if not args.context_lengths or any(length < 3 * N_PAIRS for length in args.context_lengths):
        raise ValueError(f"Every context length must be at least {3 * N_PAIRS}")
    args.context_lengths = sorted(set(args.context_lengths))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Probe: d_model={D_MODEL}, layers={N_LAYERS}, "
        f"training_steps={TRAIN_STEPS}, batch_size={BATCH_SIZE}"
    )

    accuracies: Dict[str, Dict[str, float]] = {}
    for index, variant in enumerate(VARIANTS):
        print(f"\n[INFO] Training {variant}")
        torch.manual_seed(args.seed + index)
        model = RecallProbe(variant).to(device)
        train_rng = random.Random(args.seed + index)
        train_probe(model, device, train_rng)

        print(f"[INFO] Evaluating {variant}")
        eval_rng = random.Random(args.seed + 100 + index)
        accuracies[variant] = evaluate_probe(
            model, args.context_lengths, args.n_trials, eval_rng, device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "task": "Synthetic Associative Recall (5 key/value pairs)",
        "probe": {
            "d_model": D_MODEL,
            "layers": N_LAYERS,
            "training_steps_per_variant": TRAIN_STEPS,
            "training_context_length": TRAIN_CONTEXT_LENGTH,
            "training_batch_size": BATCH_SIZE,
        },
        "n_trials_per_context": args.n_trials,
        "context_lengths": args.context_lengths,
        "accuracies": accuracies,
    }
    save_results(result, args.output_dir)
    save_plot(accuracies, args.context_lengths, args.output_dir)
    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
