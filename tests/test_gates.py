import numpy as np

from qsim.gates import HADAMARD
from qsim.state import ONE_STATE, ZERO_STATE


def test_hadamard_maps_zero_state_to_plus_state():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    transformed_state = HADAMARD @ ZERO_STATE

    np.testing.assert_allclose(transformed_state, plus_state)


def test_hadamard_maps_one_state_to_minus_state():
    minus_state = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    transformed_state = HADAMARD @ ONE_STATE

    np.testing.assert_allclose(transformed_state, minus_state)