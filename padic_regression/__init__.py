"""
p-adic Linear Regression Library

A fast algorithm for linear regression using p-adic distance metrics.
Uses tournament-based optimization for O(k) complexity instead of O(2^k).
"""

from .core import (
    DataPoint,
    padic_valuation,
    valuation_sum,
    bits_needed,
)

from .tournament import (
    tournament_winner,
    run_tournament,
    find_optimal,
    find_optimal_monte_carlo,
    brute_force_optimal,
    analyze_tournament,
)

from .multidim import (
    MultiDimDataPoint,
    multidim_valuation_sum,
    brute_force_multidim,
    coordinate_descent_tournament,
    joint_tournament_multidim,
    monte_carlo_multidim,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "DataPoint",
    "padic_valuation",
    "valuation_sum",
    "bits_needed",
    # Tournament algorithm (1D)
    "tournament_winner",
    "run_tournament",
    "find_optimal",
    "find_optimal_monte_carlo",
    "brute_force_optimal",
    "analyze_tournament",
    # Multi-dimensional
    "MultiDimDataPoint",
    "multidim_valuation_sum",
    "brute_force_multidim",
    "coordinate_descent_tournament",
    "joint_tournament_multidim",
    "monte_carlo_multidim",
]
