# Quantum Circuit Simulator

An early-stage state-vector quantum circuit simulator implemented from first principles in Python using NumPy.

I am building this project to deepen my understanding of quantum computation while developing my software-engineering skills through testing, version control, and incremental implementation. The simulator implements the underlying linear algebra directly rather than relying on an existing quantum-computing framework.

## Status - Early development

Currently implemented:

- Computational basis states |0⟩ and |1⟩ using complex NumPy state vectors
- State normalisation checking
- Tests for basis-state values, normalisation, and orthogonality
- Normalisation tests for equal-superposition and unnormalised states
- Test suite using pytest
- Linting with Ruff

## Planned

The project will be developed incrementally, with planned features including:

- General multi-qubit state vectors
- Tensor-product state construction
- Quantum gates and unitary operations
- Controlled operations
- Quantum circuits
- Measurement using the Born rule
- Entanglement and Bell states

## Approach

The core simulator is implemented directly using Python and NumPy, without using an existing quantum-computing framework to perform the simulation. Existing frameworks may later be used to verify results.