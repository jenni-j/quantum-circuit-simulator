import numpy as np

from qsim.state import ONE_STATE, ZERO_STATE


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
