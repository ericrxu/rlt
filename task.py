"""Synthetic operation sequences and their state after each operation."""

import numpy as np


BOS = 2
FIVE_STATE_TRANSITIONS = ((1, 0, 2, 3, 4), (1, 2, 3, 4, 0))


def generate_parity(n_ops: int, n_programs: int, seed: int):
    """Generate binary operations and the running parity starting from zero."""
    rng = np.random.default_rng(seed)
    operations = rng.integers(0, 2, size=(n_programs, n_ops))
    tokens = np.empty((n_programs, n_ops + 1), dtype=int)
    tokens[:, 0] = BOS
    tokens[:, 1:] = operations

    labels = np.empty((n_programs, n_ops), dtype=int)
    state = np.zeros(n_programs, dtype=int)
    for step in range(n_ops):
        state = np.bitwise_xor(state, operations[:, step])
        labels[:, step] = state
    return tokens, labels


def generate_five_state(n_ops: int, n_programs: int, seed: int):
    """Generate binary operations and apply the specified state permutations."""
    rng = np.random.default_rng(seed)
    operations = rng.integers(0, 2, size=(n_programs, n_ops))
    tokens = np.empty((n_programs, n_ops + 1), dtype=int)
    tokens[:, 0] = BOS
    tokens[:, 1:] = operations

    transitions = np.asarray(FIVE_STATE_TRANSITIONS)
    labels = np.empty((n_programs, n_ops), dtype=int)
    state = np.zeros(n_programs, dtype=int)
    for step in range(n_ops):
        state = transitions[operations[:, step], state]
        labels[:, step] = state
    return tokens, labels
