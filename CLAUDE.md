# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Qiskit to explore quantum acceleration of p-adic machine learning problems, specifically linear regression in p-adic number systems.

## Development Setup

This project uses `uv` for dependency management. Run scripts with:

```bash
uv run python <script.py>
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Dependencies

- qiskit - IBM's quantum computing framework
- qiskit-aer - Qiskit simulator backend
- matplotlib - For circuit visualization
- pylatexenc - LaTeX encoding for circuit diagrams

## Code Architecture

### Core Modules

- **padic_regression.py** - Main algorithm implementation
  - `classical_*` functions: Reference implementations for testing
  - `quantum_padic_regression_ladder()`: The ladder algorithm (classical simulation)
  - `quantum_round1_circuit()`: Quantum Grover-based search for round 1
  - `QuantumPadicRegression`: Class for building quantum circuits

- **twoadic.py** - 2-adic valuation (trailing zeros) computation
  - `count_trailing_zeros_inplace()`: Quantum circuit to count trailing zeros
  - `increment_by_one_if()`: Conditional increment with multi-qubit control
  - `stop_if_bit_is_1()`: Helper for the counting loop

- **increment_by_one_no_control.py** - Quantum ripple-carry increment
  - `increment_by_one_no_control()`: Add 1 to a quantum register

- **initialise.py** - Utility functions
  - `number_of_bits_required()`: Calculate bits needed for a value
  - `initialise_from_int()`: Load classical int into quantum register

### Running the Main Demo

```bash
uv run python padic_regression.py
```

### Running Tests

```bash
uv run python increment_by_one_no_control.py  # Test increment circuit
uv run python twoadic.py                       # Test trailing zeros
```

## Algorithm Overview

See `ALGORITHM_SUMMARY.md` for detailed explanation of:
- Classical brute-force algorithm: O(r^(n+2))
- Quantum ladder algorithm: O(n k p^2)

## CI/CD

GitHub Actions runs on push/PR to main. Note: CI config references old `requirements.txt` which has been replaced by `pyproject.toml`.

## Code Standards (from Sweep AI config)

- All new business logic should have corresponding unit tests
- Refactor large functions to be more modular
- Add docstrings to all functions and file headers
