import numpy as np

ZERO_STATE = np.array([1, 0], dtype=np.complex128)
ONE_STATE = np.array([0, 1], dtype=np.complex128)


def is_normalised(state):
    probability_sum = np.sum(np.abs(state) ** 2)

    return np.isclose(probability_sum, 1.0)


def measurement_probabilities(state):
    probabilities = np.abs(state) ** 2

    return probabilities


def tensor_product(state_a, state_b):
    return np.kron(state_a, state_b)


def marginal_probabilities(state, qubit_index):
    if qubit_index not in (0, 1):
        raise ValueError("qubit_index must be 0 or 1")

    probabilities = measurement_probabilities(state)

    if qubit_index == 0:
        return np.array(
            [
                probabilities[0] + probabilities[1],
                probabilities[2] + probabilities[3],
            ]
        )
    elif qubit_index == 1:
        return np.array(
            [
                probabilities[0] + probabilities[2],
                probabilities[1] + probabilities[3],
            ]
        )


def collapse_state(state, qubit_index, outcome):
    if qubit_index not in (0, 1):
        raise ValueError("qubit_index must be 0 or 1")

    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")

    collapsed_state = state.copy()

    if qubit_index == 0:
        if outcome == 0:
            collapsed_state[2] = 0
            collapsed_state[3] = 0
        elif outcome == 1:
            collapsed_state[0] = 0
            collapsed_state[1] = 0
    elif qubit_index == 1:
        if outcome == 0:
            collapsed_state[1] = 0
            collapsed_state[3] = 0
        elif outcome == 1:
            collapsed_state[0] = 0
            collapsed_state[2] = 0

    norm = np.linalg.norm(collapsed_state)

    if np.isclose(norm, 0.0):
        raise ValueError("collapse onto an impossible measurement outcome")

    return collapsed_state / norm


def measure_qubit(state, qubit_index):
    probabilities = marginal_probabilities(state, qubit_index)

    outcome = np.random.choice([0, 1], p=probabilities)

    collapsed_state = collapse_state(state, qubit_index, outcome)

    return outcome, collapsed_state
