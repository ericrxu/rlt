"""Contracts for next-token and per-operation state-label objectives."""

import math

import pytest
import torch

from model import RLTModel
from objectives import (
    lm_loss,
    sft_loss,
    state_tracking_accuracy,
    state_tracking_loss,
)


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(9281)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


def test_lm_loss_matches_per_example_hand_computation():
    logits = torch.tensor([
        [[0.2, -0.3, 1.1], [0.5, 0.1, -0.2], [-0.4, 0.8, 0.3], [0.1, 0.2, 0.3]],
        [[-0.7, 0.4, 0.9], [0.3, -0.2, 0.6], [1.2, 0.1, -0.5], [0.8, 0.4, -0.2]],
    ])
    tokens = torch.tensor([[2, 1, 0, 2], [2, 0, 2, 1]])

    per_example = []
    for batch in range(2):
        terms = []
        for position in range(3):
            row = logits[batch, position].tolist()
            target = tokens[batch, position + 1].item()
            terms.append(math.log(sum(math.exp(value) for value in row)) - row[target])
        per_example.append(sum(terms) / 3)
    expected = sum(per_example) / 2
    torch.testing.assert_close(lm_loss(logits, tokens), torch.tensor(expected),
                               rtol=0, atol=1e-12)


def test_lm_logits_predict_the_next_token():
    tokens = torch.tensor([[2, 0, 1, 0, 1]])
    next_token_logits = torch.zeros(1, 5, 3)
    current_token_logits = torch.zeros_like(next_token_logits)
    next_token_logits[:, :-1].scatter_(-1, tokens[:, 1:].unsqueeze(-1), 40.0)
    current_token_logits[:, :-1].scatter_(-1, tokens[:, :-1].unsqueeze(-1), 40.0)

    assert lm_loss(next_token_logits, tokens).item() < 1e-6
    assert lm_loss(current_token_logits, tokens).item() > 1.0


def test_sft_normalizes_each_example_before_batch_average():
    tokens = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
    logits = torch.zeros(2, 4, 2)
    logits[0, 0] = torch.tensor([0.0, 4.0])
    logits[1, :3, 0] = 4.0
    mask = torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)

    hard_loss = math.log1p(math.exp(4.0))
    easy_loss = math.log1p(math.exp(-4.0))
    per_example_mean = (hard_loss + easy_loss) / 2
    token_weighted_mean = (hard_loss + 3 * easy_loss) / 4
    assert abs(per_example_mean - token_weighted_mean) > 0.5
    torch.testing.assert_close(sft_loss(logits, tokens, mask),
                               torch.tensor(per_example_mean), rtol=0, atol=1e-12)


def test_sft_excludes_examples_without_selected_targets():
    logits = torch.randn(2, 4, 3)
    tokens = torch.tensor([[2, 0, 1, 2], [2, 1, 0, 1]])
    mask = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 1]], dtype=torch.bool)
    torch.testing.assert_close(sft_loss(logits, tokens, mask),
                               sft_loss(logits[1:], tokens[1:], mask[1:]),
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        sft_loss(logits, tokens, torch.zeros_like(mask))


def test_sft_rejects_bos_as_a_target():
    logits = torch.randn(2, 4, 3)
    tokens = torch.tensor([[2, 0, 1, 0], [2, 1, 0, 1]])
    mask = torch.tensor([[1, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError):
        sft_loss(logits, tokens, mask)


def test_sft_final_target_reaches_earlier_unique_token_and_initial_state():
    model = RLTModel(
        vocab_size=5, dim=8, num_encoder_layers=2, num_decoder_layers=2,
        num_heads=2, head_dim=4, window_size=3, alpha=0.75,
        rms_eps=1e-6, bos_token_id=2, num_memory_groups=1,
    )
    tokens = torch.tensor([[2, 3, 0, 1, 4, 0]])
    assert torch.count_nonzero(tokens == tokens[0, 1]) == 1
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -1] = True

    loss = sft_loss(model(tokens), tokens, mask)
    initial_grad, embedding_grad = torch.autograd.grad(
        loss, (model.s_star, model.encoder.embedding.weight)
    )
    assert torch.count_nonzero(initial_grad) > 0
    assert torch.count_nonzero(embedding_grad[tokens[0, 1]]) > 0


def test_state_tracking_scores_after_each_operation():
    labels = torch.tensor([[0, 1, 2, 0]])
    logits = torch.zeros(1, 5, 3)
    logits[:, 1:].scatter_(-1, labels.unsqueeze(-1), 40.0)
    shifted = torch.zeros_like(logits)
    shifted[:, 1:] = logits[:, 1:].roll(1, dims=1)

    assert state_tracking_loss(logits, labels).item() < 1e-6
    assert state_tracking_loss(shifted, labels).item() > 1.0


def test_state_tracking_accuracy_counts_positions_and_final_states():
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    predictions = torch.tensor([[0, 1, 0], [2, 0, 0]])
    logits = torch.zeros(2, 4, 3)
    logits[:, 1:].scatter_(-1, predictions.unsqueeze(-1), 40.0)
    per_position, final_state = state_tracking_accuracy(logits, labels)

    torch.testing.assert_close(per_position, torch.tensor(2 / 3), rtol=0, atol=1e-12)
    torch.testing.assert_close(final_state, torch.tensor(1 / 2), rtol=0, atol=1e-12)


@pytest.mark.parametrize("objective", [state_tracking_loss, state_tracking_accuracy])
def test_state_tracking_rejects_wrong_label_length(objective):
    logits = torch.randn(2, 5, 3)
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    with pytest.raises(ValueError):
        objective(logits, labels)


@pytest.mark.parametrize("objective", [state_tracking_loss, state_tracking_accuracy])
@pytest.mark.parametrize("invalid", [-1, 3, 4])
def test_state_tracking_rejects_labels_outside_readout_classes(objective, invalid):
    # Five-state labels cannot be scored by a three-class readout.
    logits = torch.randn(1, 5, 3)
    labels = torch.tensor([[0, 1, invalid, 2]])
    with pytest.raises(ValueError):
        objective(logits, labels)
