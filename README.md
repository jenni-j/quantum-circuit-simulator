# Quantum Circuit Simulator

An early-stage state-vector quantum circuit simulator implemented from first principles in Python using NumPy.

I am building this project to deepen my understanding of quantum computation while developing my software-engineering skills through testing, version control, and incremental implementation. The simulator implements the underlying linear algebra directly rather than relying on an existing quantum-computing framework.

Currently implemented:

- Computational basis states |0⟩ and |1⟩ using complex NumPy state vectors
- State normalisation checking
- Born-rule measurement probabilities
- Hadamard, Pauli-X, Pauli-Z, and CNOT gates using NumPy arrays
- Tensor-product construction of multi-qubit states
- Creation and testing of Bell states and entanglement
- Two-qubit marginal probabilities
- Single-qubit measurement within two-qubit states using probabilistic sampling
- Post-measurement state collapse
- Tests covering basis states, normalisation, gates, tensor products, entanglement, and measurement
- Testing with pytest and linting/formatting with Ruff

## Planned

The project will be developed incrementally, with planned features including:

- General n-qubit gate application
- Quantum circuit representation and execution
- Generalisation of measurement beyond the current two-qubit implementation
- Implementation of simple quantum algorithms
- Verification against an established quantum-computing framework such as Qiskit

## Approach

The core simulator is implemented directly using Python and NumPy, without using an existing quantum-computing framework to perform the simulation. Existing frameworks may later be used to verify results.