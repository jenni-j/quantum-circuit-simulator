import numpy as np

ZERO_STATE = np.array([1, 0], dtype=np.complex128)
ONE_STATE = np.array([0, 1], dtype=np.complex128)

def is_normalised(state):
    probability_sum = np.sum(np.abs(state) ** 2)

    return np.isclose(probability_sum, 1.0)

def measurement_probabilities(state):
    probabilities = np.abs(state) ** 2

    return probabilities