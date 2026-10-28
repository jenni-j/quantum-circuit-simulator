import numpy as np

HADAMARD = (1 / np.sqrt(2)) * np.array(
    [
        [1, 1],
        [1, -1],
    ], 
    dtype=np.complex128,
)

PAULI_X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=np.complex128,
)

PAULI_Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=np.complex128,
)