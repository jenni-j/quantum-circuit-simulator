import numpy as np

from qsim.state import (
    ONE_STATE,
    ZERO_STATE,
    is_normalised,
    marginal_probabilities,
    measurement_probabilities,
    tensor_product,
)


def test_zero_state():
    np.testing.assert_array_equal(
        ZERO_STATE,
        np.array([1, 0], dtype=np.complex128),
    )


def test_zero_state_is_normalised():
    probability_sum = np.sum(np.abs(ZERO_STATE) ** 2)

    assert np.isclose(probability_sum, 1.0)


def test_one_state():
    np.testing.assert_array_equal(
        ONE_STATE,
        np.array([0, 1], dtype=np.complex128),
    )


def test_one_state_is_normalised():
    probability_sum = np.sum(np.abs(ONE_STATE) ** 2)

    assert np.isclose(probability_sum, 1.0)


def test_basis_states_are_orthogonal():
    inner_product = np.vdot(ZERO_STATE, ONE_STATE)

    assert np.isclose(inner_product, 0.0)


def test_is_normalised_rejects_unnormalised_state():
    unnormalised_state = np.array([1, 1])

    assert not is_normalised(unnormalised_state)


def test_is_normalised_accepts_equal_superposition():
    equal_superposition_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)]
    )

    assert is_normalised(equal_superposition_state)


def test_measurement_probabilities_for_zero_state():
    np.testing.assert_array_equal(
        measurement_probabilities(ZERO_STATE),
        np.array([1.0, 0.0])
    )


def test_measurement_probabilities_for_equal_superposition():
    equal_superposition_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        measurement_probabilities(equal_superposition_state),
        np.array([0.5, 0.5]),
    )


def test_relative_phase_does_not_change_computational_basis_probabilities():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)], 
        dtype=np.complex128
        )
    
    minus_state = np.array(
        [1 / np.sqrt(2), -1 / np.sqrt(2)], 
        dtype=np.complex128
        )

    plus_state_probabilities = measurement_probabilities(plus_state)
    minus_state_probabilities = measurement_probabilities(minus_state)

    np.testing.assert_allclose(plus_state_probabilities, minus_state_probabilities)


def test_measurement_probabilities_for_complex_phase_state():
    complex_phase_state = np.array(
        [1 / np.sqrt(2), 1j / np.sqrt(2)],
        dtype=np.complex128,
    )

    complex_phase_state_probabilities = measurement_probabilities(complex_phase_state)

    np.testing.assert_allclose(
        complex_phase_state_probabilities,
        np.array([0.5, 0.5]),
    )


def test_tensor_product_of_zero_state_and_one_state():
    result = tensor_product(ZERO_STATE, ONE_STATE)

    np.testing.assert_allclose(
        result,
        np.array(
            [0, 1, 0, 0],
            dtype=np.complex128,
        ),
    )


def test_tensor_product_of_plus_state_and_zero_state():
    plus_state = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)], 
        dtype=np.complex128
    )

    result = tensor_product(plus_state, ZERO_STATE)

    np.testing.assert_allclose(
        result,
        (1 / np.sqrt(2))
        * np.array(
            [1, 0, 1, 0],
            dtype=np.complex128,
        ),
    )


def test_measurement_probabilities_for_bell_state():
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    bell_state = (1 / np.sqrt(2)) * (zero_zero_state + one_one_state)

    result = measurement_probabilities(bell_state)

    np.testing.assert_allclose(
        result,
        np.array([0.5, 0, 0, 0.5]),
    )


def test_first_qubit_marginal_probabilities_for_bell_state():
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    bell_state = (1 / np.sqrt(2)) * (zero_zero_state + one_one_state)

    probabilities = measurement_probabilities(bell_state)

    probability_first_qubit_zero = probabilities[0] + probabilities[1]
    probability_first_qubit_one = probabilities[2] + probabilities[3]

    np.testing.assert_allclose(
        probability_first_qubit_zero,
        0.5,
    )

    np.testing.assert_allclose(
        probability_first_qubit_one,
        0.5,
    )


def test_first_qubit_marginal_probabilities_for_zero_one_state():
    zero_one_state = tensor_product(ZERO_STATE, ONE_STATE)

    probabilities = marginal_probabilities(zero_one_state, 0)

    np.testing.assert_allclose(
        probabilities,
        np.array([1.0, 0.0]),
    )


def test_second_qubit_marginal_probabilities_for_zero_one_state():
    zero_one_state = tensor_product(ZERO_STATE, ONE_STATE)

    probabilities = marginal_probabilities(zero_one_state, 1)

    np.testing.assert_allclose(
        probabilities,
        np.array([0.0, 1.0]),
    )


def test_marginal_probabilities_for_bell_state():
    zero_zero_state = tensor_product(ZERO_STATE, ZERO_STATE)
    one_one_state = tensor_product(ONE_STATE, ONE_STATE)
    bell_state = (1 / np.sqrt(2)) * (zero_zero_state + one_one_state)

    probabilities_first_qubit = marginal_probabilities(bell_state, 0)
    probabilities_second_qubit = marginal_probabilities(bell_state, 1)

    np.testing.assert_allclose(
        probabilities_first_qubit,
        np.array([0.5, 0.5]),
    )

    np.testing.assert_allclose(
        probabilities_second_qubit,
        np.array([0.5, 0.5]),
    )