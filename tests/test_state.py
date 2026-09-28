import numpy as np

from qsim.state import ONE_STATE, ZERO_STATE, is_normalised, measurement_probabilities


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