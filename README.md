# p-adic Linear Regression

A Python library for linear regression using p-adic (specifically 2-adic) distance metrics.

## Background

This project began as an exploration of quantum acceleration for p-adic machine learning, based on the paper's ladder algorithm. Through experimentation, we discovered that the **tournament structure itself provides exponential speedup classically** - no quantum mechanics required!

The key insight: p-adic regression can be solved via coordinate-wise bit optimization, where each bit of the optimal coefficient is determined by a single pairwise comparison.

## The Algorithm

### Tournament-Based Optimization

For k-bit coefficients, instead of searching all 2^k possibilities:

1. Start with any coefficient
2. Run k rounds of tournament:
   - Round i: Compare current coefficient `a` vs `a XOR 2^(i-1)`
   - Keep the winner (higher p-adic valuation sum)
3. Result: optimal coefficient in O(k) comparisons

### Complexity

| Approach | Evaluations | Example (12-bit) |
|----------|-------------|------------------|
| Brute Force | O(2^k) | 4,096 |
| Tournament | O(k) | 24 |
| **Speedup** | **Exponential** | **171x** |

## Installation

```bash
# Clone the repository
git clone https://github.com/solresol/quantum-padic-linear-regression.git
cd quantum-padic-linear-regression

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```python
from padic_regression import DataPoint, tournament_regression

# Your data points
data = [
    DataPoint(1, 5),
    DataPoint(2, 10),
    DataPoint(3, 15)
]

# Find optimal coefficient (y = a*x through origin)
optimal_a = tournament_regression(data, coeff_bits=4)
print(f"Optimal coefficient: {optimal_a}")
```

## Usage

### Basic Regression (Line Through Origin)

```python
from classical_monte_carlo import monte_carlo_ladder, brute_force_optimal
from padic_core import DataPoint

data = [DataPoint(1, 3), DataPoint(2, 6), DataPoint(4, 12)]

# Tournament approach (fast)
winner, counts = monte_carlo_ladder(data, coeff_bits=4, n_samples=1)

# Verify with brute force
optimal, valuation = brute_force_optimal(data, coeff_bits=4)

print(f"Tournament: {winner}, Brute force: {optimal}")
```

### Understanding the Results

The algorithm minimizes p-adic distance, which measures how divisible residuals are by 2:
- Residual = 0: infinite valuation (perfect fit)
- Residual = 8: valuation = 3 (divisible by 2^3)
- Residual = 6: valuation = 1 (divisible by 2^1)

Higher total valuation = better fit in p-adic sense.

## Project Structure

### Core Library

- **padic_core.py** - Core p-adic functions
  - `DataPoint`: Data point class
  - `classical_2adic_valuation()`: Count trailing zeros
  - `classical_valuation_sum()`: Total valuation for a fit

- **classical_monte_carlo.py** - Main algorithm
  - `tournament_winner()`: Compare two coefficients
  - `run_single_tournament()`: Full tournament from starting point
  - `monte_carlo_ladder()`: Monte Carlo with multiple samples
  - `brute_force_optimal()`: Reference implementation

### Quantum Experiments (Historical)

The `quantum_*.py` files contain our quantum computing experiments using Qiskit. These were instrumental in understanding the algorithm but are not needed for the classical solution.

## Development

```bash
# Run tests
uv run python classical_monte_carlo.py

# Run experiments
uv run python experiments.py
```

## Related Work

- [pyadic](https://github.com/GDeLaurentis/pyadic) - p-adic number types for Python (complementary library)
- Original quantum ladder algorithm paper (TODO: add citation)

## License

MIT

## Contributing

Contributions welcome! Areas of interest:
- Multi-dimensional regression (multiple coefficients)
- Different primes (3-adic, 5-adic, etc.)
- Integration with pyadic number types
- Performance optimizations
