import numpy as np

from qsim.gates import CNOT, HADAMARD, PAULI_X, PAULI_Z
from qsim.state import ONE_STATE, ZERO_STATE, tensor_product


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


def test_pauli_x_maps_zero_state_to_one_state():
    result = PAULI_X @ ZERO_STATE

    np.testing.assert_allclose(
        result,
        ONE_STATE,
    )


def test_pauli_x_maps_one_state_to_zero_state():
    result = PAULI_X @ ONE_STATE

    np.testing.assert_allclose(
        result,
        ZERO_STATE,
    )


def test_pauli_z_maps_zero_state_to_zero_state():
    result = PAULI_Z @ ZERO_STATE

    np.testing.assert_allclose(
        result,
        ZERO_STATE,
    )


def test_pauli_z_maps_one_state_to_negative_one_state():
    result = PAULI_Z @ ONE_STATE

    np.testing.assert_allclose(
        result,
        -ONE_STATE,
    )


def test_pauli_z_maps_plus_state_to_minus_state():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    minus_state = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    result = PAULI_Z @ plus_state

    np.testing.assert_allclose(
        result,
        minus_state,
    )


def test_pauli_z_maps_minus_state_to_plus_state():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    minus_state = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    result = PAULI_Z @ minus_state

    np.testing.assert_allclose(
        result,
        plus_state,
    )


def test_cnot_maps_zero_zero_state_to_zero_zero_state():
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)

    result = CNOT @ zero_zero_state

    np.testing.assert_allclose(
        result,
        zero_zero_state,
    )


def test_cnot_maps_zero_one_state_to_zero_one_state():
    zero_one_state = tensor_product(ZERO_STATE, ONE_STATE)

    result = CNOT @ zero_one_state

    np.testing.assert_allclose(
        result,
        zero_one_state,
    )


def test_cnot_maps_one_zero_state_to_one_one_state():
    one_zero_state = tensor_product(ONE_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)

    result = CNOT @ one_zero_state

    np.testing.assert_allclose(
        result,
        one_one_state,
    )


def test_cnot_maps_one_one_state_to_one_zero_state():
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    one_zero_state = tensor_product(ONE_STATE, ZERO_STATE)

    result = CNOT @ one_one_state

    np.testing.assert_allclose(
        result,
        one_zero_state,
    )


def test_cnot_maps_superposition_to_bell_state():
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_zero_state = tensor_product(ONE_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    superposition_state = (1 / np.sqrt(2)) * (zero_zero_state + one_zero_state)
    bell_state = (1 / np.sqrt(2)) * (zero_zero_state + one_one_state)

    result = CNOT @ superposition_state

    np.testing.assert_allclose(
        result,
        bell_state,
    )


def test_hadamard_on_first_qubit_creates_two_qubit_superposition():
    hadamard_first_qubit = np.kron(HADAMARD, np.eye(2))
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_zero_state = tensor_product(ONE_STATE, ZERO_STATE)

    result = hadamard_first_qubit @ zero_zero_state

    np.testing.assert_allclose(
        result,
        (1 / np.sqrt(2)) * (zero_zero_state + one_zero_state),
    )


def test_hadamard_then_cnot_creates_bell_state():
    hadamard_first_qubit = np.kron(HADAMARD, np.eye(2))
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    superposition_state = hadamard_first_qubit @ zero_zero_state

    result = CNOT @ superposition_state

    np.testing.assert_allclose(
        result,
        (1 / np.sqrt(2)) * (zero_zero_state + one_one_state),
    )