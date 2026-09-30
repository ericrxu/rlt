"""Training objectives and metrics for next-token and state-label tasks."""

import torch
import torch.nn.functional as F


def _check_token_shapes(logits: torch.Tensor, tokens: torch.Tensor) -> None:
    if logits.ndim != 3 or tokens.ndim != 2:
        raise ValueError("logits must be (B, T, V) and tokens must be (B, T)")
    if logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must have matching batch and time dimensions")


def lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Equation (5.1): next-token cross entropy, normalized per sequence."""
    _check_token_shapes(logits, tokens)
    if tokens.shape[1] < 2:
        raise ValueError("language-model loss requires at least two tokens")
    predictions = logits[:, :-1, :]
    targets = tokens[:, 1:]
    per_token = F.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return per_token.mean(dim=1).mean()


def sft_loss(
    logits: torch.Tensor, tokens: torch.Tensor, assistant_mask: torch.Tensor
) -> torch.Tensor:
    """Equation (5.2): select assistant targets and average per nonempty example."""
    _check_token_shapes(logits, tokens)
    if assistant_mask.shape != tokens.shape:
        raise ValueError("assistant_mask must have the same shape as tokens")
    if tokens.shape[1] < 2:
        raise ValueError("SFT loss requires at least two tokens")
    if torch.any(assistant_mask[:, 0]):
        raise ValueError("BOS cannot be selected as an SFT target")
    targets = tokens[:, 1:]
    selected = assistant_mask[:, 1:].to(device=logits.device, dtype=torch.bool)
    per_token = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    counts = selected.sum(dim=1)
    nonempty = counts > 0
    if not torch.any(nonempty):
        raise ValueError("SFT batch must contain at least one selected target")
    per_example = (per_token * selected).sum(dim=1)[nonempty] / counts[nonempty]
    return per_example.mean()


def _state_tracking_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be (B, T, V) and labels must be (B, N)")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1] + 1:
        raise ValueError("logits must have exactly one more time position than labels")
    if labels.numel() and (torch.any(labels < 0) or torch.any(labels >= logits.shape[-1])):
        raise ValueError("labels must be within the readout class range")
    return logits[:, 1:, :]


def state_tracking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Score each operation's resulting state from readout position i+1."""
    predictions = _state_tracking_logits(logits, labels)
    return F.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]), labels.reshape(-1)
    )


def state_tracking_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position-weighted accuracy and accuracy on final states."""
    predictions = _state_tracking_logits(logits, labels).argmax(dim=-1)
    correct = predictions == labels
    per_position = correct.to(dtype=logits.dtype).mean()
    final_state = correct[:, -1].to(dtype=logits.dtype).mean()
    return per_position, final_state
