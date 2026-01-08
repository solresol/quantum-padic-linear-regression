# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements p-adic linear regression using a tournament-based optimization algorithm.

**History**: Originally explored quantum acceleration for p-adic ML. We discovered the tournament structure provides exponential speedup classically - no quantum needed! The quantum code remains for historical reference.

## Development Setup

This project uses `uv` for dependency management. Run scripts with:

```bash
uv run python <script.py>
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Core Algorithm

### Tournament-Based Optimization

For k-bit coefficients, finds optimal in O(k) evaluations instead of O(2^k):

1. Start from any coefficient
2. For each bit position i (0 to k-1):
   - Compare current `a` with `a XOR 2^i`
   - Keep the one with higher valuation sum
3. Result: optimal coefficient

**Key insight**: Each bit can be optimized independently via pairwise comparison.

## Code Architecture

### Primary Modules (Classical)

- **padic_core.py** - Core p-adic functions
  - `DataPoint`: Data point class (x, y coordinates)
  - `classical_2adic_valuation(n)`: Count trailing zeros in binary
  - `classical_valuation_sum(data, m, b)`: Total valuation for line y=mx+b
  - `bits_needed(max_val)`: Calculate required bits

- **classical_monte_carlo.py** - Main algorithm (RECOMMENDED)
  - `tournament_winner(data, a, a')`: Compare two coefficients
  - `run_single_tournament(data, coeff_bits, start_a)`: Full tournament
  - `monte_carlo_ladder(data, coeff_bits, n_samples)`: Monte Carlo approach
  - `brute_force_optimal(data, coeff_bits)`: Reference implementation
  - `analyze_tournament_structure(data, coeff_bits)`: Debug/analysis tool

### Historical Modules (Quantum)

- **quantum_ladder.py**, **quantum_ladder_full.py** - Quantum implementations
- **quantum_oracle.py** - Grover-based search
- **quantum_arithmetic.py** - Quantum arithmetic circuits
- **twoadic.py** - Quantum trailing zeros counter
- Other quantum_*.py files

### Running Tests

```bash
# Main algorithm tests
uv run python classical_monte_carlo.py

# Quantum experiments (historical)
uv run python experiments.py
```

## Algorithm Complexity

| Approach | Evaluations | 12-bit Example |
|----------|-------------|----------------|
| Brute Force | O(2^k) | 4,096 |
| Tournament | O(2k) | 24 |
| Speedup | Exponential | 171x |

## Current Limitations

1. **Single coefficient**: Only fits y = a*x (through origin)
2. **2-adic only**: Hardcoded for prime p=2
3. **Integer data**: Requires integer x, y values

## Future Work

- Multi-dimensional regression (y = a₁x₁ + a₂x₂ + ... + b)
- Support for other primes (3-adic, 5-adic)
- Integration with pyadic library for number types
- Proper package structure for pip installation

## CI/CD

GitHub Actions runs on push/PR to main.

## Code Standards

- All new business logic should have corresponding unit tests
- Refactor large functions to be more modular
- Add docstrings to all functions and file headers
