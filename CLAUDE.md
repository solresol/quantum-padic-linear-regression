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

- **padic_regression.py** - Main algorithm with demo
  - `classical_brute_force()`: Reference implementation
  - `quantum_padic_regression_ladder()`: Ladder algorithm (classical sim)
  - Integrates `quantum_find_optimal()` for full quantum search

- **padic_core.py** - Shared functions and classes
  - `DataPoint`: Data point class
  - `classical_2adic_valuation()`, `classical_valuation_sum()`: Reference implementations
  - `bits_needed()`: Utility function

- **quantum_oracle.py** - Quantum Grover search
  - `QuantumResidualOracle`: Builds oracle circuits for residual computation
  - `quantum_find_optimal()`: Full Grover search for optimal (m, b)
  - `build_grover_oracle()`: Creates phase oracle for given threshold

- **quantum_arithmetic.py** - Quantum arithmetic circuits
  - `quantum_add()`: Ripple-carry addition
  - `quantum_subtract()`: Two's complement subtraction
  - `multiply_by_constant()`: Multiply quantum register by classical value
  - `controlled_add_classical()`: Controlled addition of classical constant

- **twoadic.py** - 2-adic valuation (trailing zeros)
  - `count_trailing_zeros_inplace()`: Quantum trailing zeros counter
  - `increment_by_one_if()`: Conditional increment
  - `stop_if_bit_is_1()`: Helper for counting loop

- **increment_by_one_no_control.py** - Quantum increment
  - `increment_by_one_no_control()`: Add 1 to quantum register

- **initialise.py** - Utilities
  - `number_of_bits_required()`: Calculate bits needed
  - `initialise_from_int()`: Load int into quantum register

### Running the Main Demo

```bash
uv run python padic_regression.py
```

### Running Tests

```bash
uv run python increment_by_one_no_control.py  # Test increment circuit
uv run python twoadic.py                       # Test trailing zeros
uv run python quantum_arithmetic.py            # Test add/subtract/multiply
uv run python quantum_oracle.py                # Test Grover search
uv run python test1.py                         # GHZ state demo
uv run python test2.py                         # Grover's algorithm demo
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
