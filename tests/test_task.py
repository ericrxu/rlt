"""Contract tests for the two synthetic program generators."""

import numpy as np
import pytest

from task import BOS, FIVE_STATE_TRANSITIONS, generate_five_state, generate_parity


N_PROGRAMS = 10_000
LENGTHS = (8, 32, 128)
EXPECTED_TRANSITIONS = ((1, 0, 2, 3, 4), (1, 2, 3, 4, 0))
GENERATORS = (generate_parity, generate_five_state)


@pytest.mark.parametrize("n_ops", LENGTHS)
def test_parity_labels_at_every_step(n_ops):
    tokens, labels = generate_parity(n_ops, N_PROGRAMS, seed=1000 + n_ops)
    assert tokens.shape == (N_PROGRAMS, n_ops + 1)
    assert labels.shape == (N_PROGRAMS, n_ops)

    expected = np.empty((N_PROGRAMS, n_ops), dtype=int)
    for row, program in enumerate(tokens):
        parity = 0
        for step, operation in enumerate(program[1:]):
            if operation == 1:
                parity = 1 - parity
            expected[row, step] = parity

    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("n_ops", LENGTHS)
def test_five_state_labels_at_every_step(n_ops):
    tokens, labels = generate_five_state(n_ops, N_PROGRAMS, seed=2000 + n_ops)
    assert tokens.shape == (N_PROGRAMS, n_ops + 1)
    assert labels.shape == (N_PROGRAMS, n_ops)

    expected = np.empty((N_PROGRAMS, n_ops), dtype=int)
    for row, program in enumerate(tokens):
        state = 0
        for step, operation in enumerate(program[1:]):
            state = EXPECTED_TRANSITIONS[int(operation)][state]
            expected[row, step] = state

    np.testing.assert_array_equal(labels, expected)


def test_five_state_transition_table_is_the_specified_permutations():
    table = np.asarray(FIVE_STATE_TRANSITIONS)
    np.testing.assert_array_equal(table, EXPECTED_TRANSITIONS)
    for transition in table:
        assert sorted(transition.tolist()) == list(range(5))


def test_five_state_operations_do_not_commute():
    table = np.asarray(FIVE_STATE_TRANSITIONS)
    assert any(
        table[1, table[0, state]] != table[0, table[1, state]]
        for state in range(5)
    )


@pytest.mark.parametrize("generate", GENERATORS)
def test_same_seed_reproduces_programs_and_labels(generate):
    first_tokens, first_labels = generate(32, 256, seed=314159)
    second_tokens, second_labels = generate(32, 256, seed=314159)
    np.testing.assert_array_equal(first_tokens, second_tokens)
    np.testing.assert_array_equal(first_labels, second_labels)


@pytest.mark.parametrize("generate", GENERATORS)
def test_different_seeds_change_programs(generate):
    first_tokens, _ = generate(32, 256, seed=314159)
    second_tokens, _ = generate(32, 256, seed=271828)
    assert not np.array_equal(first_tokens, second_tokens)


@pytest.mark.parametrize("generate", GENERATORS)
def test_bos_appears_only_in_column_zero(generate):
    tokens, _ = generate(32, N_PROGRAMS, seed=42)
    assert BOS == 2
    assert BOS not in (0, 1)
    assert np.all(tokens[:, 0] == BOS)
    assert np.all((tokens[:, 1:] == 0) | (tokens[:, 1:] == 1))
    assert not np.any(tokens[:, 1:] == BOS)


@pytest.mark.parametrize("generate", GENERATORS)
def test_operations_are_sampled_uniformly(generate):
    tokens, _ = generate(32, N_PROGRAMS, seed=8675309)
    fraction_of_ones = np.mean(tokens[:, 1:] == 1)
    assert 0.49 < fraction_of_ones < 0.51
