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


def test_hadamard_gate_is_self_inverse():
    identity_matrix = np.eye(2)

    h_squared = HADAMARD @ HADAMARD

    np.testing.assert_allclose(
        h_squared, 
        identity_matrix,
        atol=1e-15,
    )


def test_hadamard_gate_is_unitary():
    conjugate_transpose = HADAMARD.conj().T
    identity_matrix = np.eye(2)

    np.testing.assert_allclose(
        HADAMARD @ conjugate_transpose,
        identity_matrix,
        atol=1e-15,
    )


def test_hadamard_maps_plus_state_to_zero_state():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    result = HADAMARD @ plus_state

    np.testing.assert_allclose(
        result,
        ZERO_STATE,
    )


def test_hadamard_maps_minus_state_to_one_state():
    minus_state = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    result = HADAMARD @ minus_state

    np.testing.assert_allclose(
        result,
        ONE_STATE,
    )