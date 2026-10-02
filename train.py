"""Deterministic training runner for the state-tracking experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from baselines import GRUBaseline, TransformerBaseline
from model import RLTModel
from objectives import final_state_loss, state_tracking_accuracy, state_tracking_loss
from task import BOS, generate_five_state, generate_parity


@dataclass(frozen=True)
class TrainConfig:
    name: str
    task: str
    model_type: str
    objective: str
    seed: int
    eval_seed: int
    dim: int
    num_encoder_layers: int
    num_decoder_layers: int | None
    num_heads: int | None
    head_dim: int | None
    window_size: int | None
    alpha: float | None
    rms_eps: float | None
    train_length: int
    batch_size: int
    train_programs: int | None
    curriculum: list[list[int]] | None
    length_mix: list[int] | None
    steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip_norm: float
    eval_lengths: tuple[int, ...]
    eval_programs: int
    eval_interval: int

    def __post_init__(self) -> None:
        if self.task not in ("parity", "five_state"):
            raise ValueError("task must be parity or five_state")
        if self.objective not in ("per_position", "final_state"):
            raise ValueError("objective must be per_position or final_state")
        unused = {
            "rlt": (),
            "transformer": ("num_decoder_layers", "window_size", "alpha"),
            "gru": (
                "num_decoder_layers", "num_heads", "head_dim", "window_size",
                "alpha", "rms_eps",
            ),
        }
        if self.model_type not in unused:
            raise ValueError("model_type must be rlt, transformer, or gru")
        model_fields = (
            "dim", "num_encoder_layers", "num_decoder_layers", "num_heads",
            "head_dim", "window_size", "alpha", "rms_eps",
        )
        for name in model_fields:
            value = getattr(self, name)
            if name in unused[self.model_type]:
                if value is not None:
                    raise ValueError(f"{name} must be null for {self.model_type}")
            elif value is None:
                raise ValueError(f"{name} is required for {self.model_type}")
        if self.train_programs is not None and self.batch_size != self.train_programs:
            raise ValueError("batch_size must equal train_programs for a fixed dataset")
        if self.curriculum is not None:
            if not isinstance(self.curriculum, list) or not self.curriculum:
                raise ValueError("curriculum must be a nonempty list of stages")
            if self.train_programs is not None:
                raise ValueError("curriculum requires train_programs to be null")
            previous_length = 0
            total_steps = 0
            for stage in self.curriculum:
                if (not isinstance(stage, list) or len(stage) != 2
                        or any(type(value) is not int or value < 1 for value in stage)):
                    raise ValueError("each curriculum stage must be [positive length, positive steps]")
                length, stage_steps = stage
                if length < previous_length:
                    raise ValueError("curriculum lengths must be non-decreasing")
                previous_length = length
                total_steps += stage_steps
            if total_steps != self.steps:
                raise ValueError("curriculum stage steps must sum to steps")
            if previous_length != self.train_length:
                raise ValueError("curriculum must end at train_length")
        if self.length_mix is not None:
            if (not isinstance(self.length_mix, list) or not self.length_mix
                    or any(type(length) is not int or length < 1
                           for length in self.length_mix)):
                raise ValueError("length_mix must be a nonempty list of positive lengths")
            if any(left >= right for left, right in zip(self.length_mix, self.length_mix[1:])):
                raise ValueError("length_mix must be strictly increasing")
            if self.length_mix[-1] != self.train_length:
                raise ValueError("length_mix must end at train_length")
            if self.batch_size % len(self.length_mix):
                raise ValueError("batch_size must be divisible by length_mix size")
            if self.train_programs is not None:
                raise ValueError("length_mix requires train_programs to be null")
            if self.curriculum is not None:
                raise ValueError("length_mix cannot be combined with curriculum")
        for name in ("dim", "num_encoder_layers", "num_decoder_layers", "num_heads",
                     "head_dim", "window_size", "train_length", "batch_size", "steps",
                     "warmup_steps", "eval_programs", "eval_interval"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if not self.eval_lengths or any(length < 1 for length in self.eval_lengths):
            raise ValueError("eval_lengths must contain positive lengths")
        if self.warmup_steps > self.steps:
            raise ValueError("warmup_steps cannot exceed steps")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("learning rates must satisfy 0 <= min <= peak")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("learning_rate and grad_clip_norm must be positive")


def load_config(path: str | Path) -> TrainConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "curriculum" not in data:
        raise ValueError("curriculum is required")
    if "length_mix" not in data:
        raise ValueError("length_mix is required")
    return TrainConfig(**data)


def build_model(config: TrainConfig) -> torch.nn.Module:
    vocab_size = 3 if config.task == "parity" else 5
    if config.model_type == "transformer":
        return TransformerBaseline(
            vocab_size, config.dim, config.num_encoder_layers,
            config.num_heads, config.head_dim, config.rms_eps,
        )
    if config.model_type == "gru":
        return GRUBaseline(vocab_size, config.dim, config.num_encoder_layers)
    return RLTModel(
        vocab_size=vocab_size,
        dim=config.dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        window_size=config.window_size,
        alpha=config.alpha,
        rms_eps=config.rms_eps,
        bos_token_id=BOS,
        num_memory_groups=1,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _task_number(task: str) -> int:
    return 0 if task == "parity" else 1


def _derived_seed(*parts: int) -> int:
    """Domain-separated, stable seed for NumPy's program generator."""
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])


def _batch(config: TrainConfig, length: int, count: int, seed: int):
    generator = generate_parity if config.task == "parity" else generate_five_state
    tokens, labels = generator(length, count, seed)
    return torch.as_tensor(tokens, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long)


def make_training_batch(config: TrainConfig, step: int):
    if step < 1:
        raise ValueError("training steps are one-based")
    length = config.train_length
    if config.curriculum is not None:
        end_step = 0
        for stage_length, stage_steps in config.curriculum:
            end_step += stage_steps
            if step <= end_step:
                length = stage_length
                break
    if config.train_programs is None:
        seed = _derived_seed(0, config.seed, _task_number(config.task), step)
        count = config.batch_size
    else:
        seed = _derived_seed(0, config.seed, _task_number(config.task))
        count = config.train_programs
    return _batch(config, length, count, seed)


def make_training_groups(config: TrainConfig, step: int):
    if config.length_mix is None:
        return [make_training_batch(config, step)]
    if step < 1:
        raise ValueError("training steps are one-based")
    count = config.batch_size // len(config.length_mix)
    return [
        _batch(config, length, count,
               _derived_seed(2, config.seed, _task_number(config.task), step, length))
        for length in config.length_mix
    ]


def make_eval_batch(config: TrainConfig, length: int):
    seed = _derived_seed(1, config.eval_seed, _task_number(config.task), length)
    return _batch(config, length, config.eval_programs, seed)


def make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
    )


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
    if not 1 <= step <= config.steps:
        raise ValueError("step must be within the training run")
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.min_learning_rate + 0.5 * (
        config.learning_rate - config.min_learning_rate
    ) * (1 + math.cos(math.pi * progress))


def clip_gradients(model: torch.nn.Module, limit: float) -> float:
    if limit <= 0:
        raise ValueError("gradient clip norm must be positive")
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), limit).item())


@dataclass
class Diagnostics:
    gate_values: list[torch.Tensor] = field(default_factory=list)
    state_norms: list[torch.Tensor] = field(default_factory=list)


@contextmanager
def diagnostic_hooks(model: torch.nn.Module):
    diagnostics = Diagnostics()

    if not isinstance(model, RLTModel):
        yield diagnostics
        return

    def record_gate(_module, _inputs, output):
        diagnostics.gate_values.append(output[1].detach())

    def record_state(_module, _inputs, output):
        diagnostics.state_norms.append(torch.linalg.vector_norm(output[0].detach(), dim=-1))

    handles = [model.merge.register_forward_hook(record_gate),
               model.decoder.register_forward_hook(record_state)]
    try:
        yield diagnostics
    finally:
        for handle in handles:
            handle.remove()


def invalid_prediction_rate(logits: torch.Tensor, task: str) -> float:
    """Fraction of operation positions predicted outside the task's label classes."""
    if task not in ("parity", "five_state"):
        raise ValueError("task must be parity or five_state")
    valid_classes = 2 if task == "parity" else 5
    predictions = logits[:, 1:, :].argmax(dim=-1)
    return (predictions >= valid_classes).sum().item() / predictions.numel()


def evaluate(model: torch.nn.Module, config: TrainConfig) -> dict[str, dict[str, float]]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results = {}
    try:
        with torch.no_grad():
            for length in config.eval_lengths:
                tokens, labels = make_eval_batch(config, length)
                position_correct = 0.0
                final_correct = 0.0
                invalid_count = 0.0
                for start in range(0, config.eval_programs, config.batch_size):
                    batch_tokens = tokens[start:start + config.batch_size].to(device)
                    batch_labels = labels[start:start + config.batch_size].to(device)
                    logits = model(batch_tokens)
                    position, final = state_tracking_accuracy(logits, batch_labels)
                    size = batch_tokens.shape[0]
                    position_correct += position.item() * size
                    final_correct += final.item() * size
                    invalid_count += invalid_prediction_rate(logits, config.task) * size
                results[str(length)] = {
                    "per_position_accuracy": position_correct / config.eval_programs,
                    "final_state_accuracy": final_correct / config.eval_programs,
                    "invalid_prediction_rate": invalid_count / config.eval_programs,
                }
    finally:
        model.train(was_training)
    return results


def _fixed_training_accuracy(model: torch.nn.Module, config: TrainConfig) -> dict[str, float]:
    tokens, labels = make_training_batch(config, 1)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(tokens.to(device))
            position, final = state_tracking_accuracy(logits, labels.to(device))
    finally:
        model.train(was_training)
    return {
        "per_position_accuracy": position.item(),
        "final_state_accuracy": final.item(),
    }


@dataclass
class TrainResult:
    model: torch.nn.Module
    loss_history: list[float]
    eval_history: list[dict]
    diagnostics: list[dict]
    results_path: Path
    checkpoint_path: Path


def _git_metadata() -> tuple[str, bool]:
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
        cwd=root, capture_output=True, text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def train(
    config: TrainConfig,
    output_dir: str | Path = "results",
    dtype: torch.dtype = torch.float32,
    checkpoint_dir: str | Path = "checkpoints",
    log_every: int | None = None,
) -> TrainResult:
    if log_every is not None and log_every < 1:
        raise ValueError("log_every must be positive")
    commit, dirty = _git_metadata()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = build_model(config).to(dtype=dtype)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda completed: learning_rate_at_step(
            config, min(completed + 1, config.steps)
        ) / config.learning_rate,
    )
    loss_history: list[float] = []
    eval_history: list[dict] = []
    diagnostics_history: list[dict] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = output_dir / f"{stamp}_{config.name}.json"
    checkpoint_path = checkpoint_dir / f"{stamp}_{config.name}.pt"
    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with diagnostic_hooks(model) as diagnostics:
            if config.length_mix is None:
                tokens, labels = make_training_batch(config, step)
                logits = model(tokens)
                if config.objective == "per_position":
                    loss = state_tracking_loss(logits, labels)
                else:
                    loss = final_state_loss(logits, labels)
            else:
                group_losses = []
                for tokens, labels in make_training_groups(config, step):
                    logits = model(tokens)
                    if config.objective == "per_position":
                        group_losses.append(state_tracking_loss(logits, labels))
                    else:
                        group_losses.append(final_state_loss(logits, labels))
                loss = torch.stack(group_losses).mean()
        loss.backward()
        grad_norm = clip_gradients(model, config.grad_clip_norm)
        step_learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        loss_history.append(float(loss.detach().item()))
        diagnostics_history.append({
            "step": step,
            "preclip_grad_norm": grad_norm,
            "mean_gate": (
                torch.stack([gate.mean() for gate in diagnostics.gate_values]).mean().item()
                if diagnostics.gate_values else None
            ),
            "mean_state_norm": (
                torch.stack([norm.mean() for norm in diagnostics.state_norms]).mean().item()
                if diagnostics.state_norms else None
            ),
        })
        if step % config.eval_interval == 0 or step == config.steps:
            entry = {"step": step, "lengths": evaluate(model, config)}
            if config.train_programs is not None:
                entry["training_set"] = _fixed_training_accuracy(model, config)
            eval_history.append(entry)
            torch.save(
                {"config": asdict(config), "step": step, "model": model.state_dict()},
                checkpoint_dir / f"{stamp}_{config.name}_step{step}.pt",
            )
        if log_every is not None and step % log_every == 0:
            mean_gate = diagnostics_history[-1]["mean_gate"]
            gate_text = "None" if mean_gate is None else f"{mean_gate:.9g}"
            print(
                f"step={step} loss={loss_history[-1]:.9g} "
                f"preclip_grad_norm={grad_norm:.9g} lr={step_learning_rate:.9g} "
                f"mean_gate={gate_text}"
            )

    payload = {
        "config": asdict(config), "git_commit": commit, "dirty": dirty,
        "seed": config.seed, "dtype": str(dtype).removeprefix("torch."),
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__, "parameter_count": count_parameters(model),
        "loss_history": loss_history, "eval_history": eval_history,
        "diagnostics": diagnostics_history,
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    torch.save({"config": asdict(config), "step": config.steps,
                "model": model.state_dict()}, checkpoint_path)
    return TrainResult(model, loss_history, eval_history, diagnostics_history,
                       results_path, checkpoint_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-every", type=int, default=None)
    arguments = parser.parse_args()
    outcome = train(
        load_config(arguments.config), output_dir=arguments.output_dir,
        checkpoint_dir=arguments.checkpoint_dir,
        log_every=arguments.log_every,
    )
    print(outcome.results_path)
