"""
Tournament-Based p-adic Regression

Implements the coordinate-wise bit optimization algorithm that finds
optimal coefficients in O(k) evaluations instead of O(2^k) brute force.

The key insight: for p-adic regression, each bit of the optimal coefficient
can be determined independently by a single pairwise comparison.
"""

import random
from collections import Counter
from typing import List, Tuple, Dict, Optional

from .core import DataPoint, valuation_sum


def tournament_winner(data: List[DataPoint], a: int, a_prime: int,
                      b: int = 0, p: int = 2) -> int:
    """
    Compare two coefficients and return the winner.

    Winner is the coefficient with higher valuation sum (better p-adic fit).
    Ties go to the smaller coefficient for determinism.

    Args:
        data: List of DataPoint objects
        a: First coefficient to compare
        a_prime: Second coefficient to compare
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        The winning coefficient
    """
    f_a = valuation_sum(data, a, b, p)
    f_a_prime = valuation_sum(data, a_prime, b, p)

    if f_a > f_a_prime:
        return a
    elif f_a_prime > f_a:
        return a_prime
    else:
        # Tie: prefer smaller coefficient for determinism
        return min(a, a_prime)


def run_tournament(data: List[DataPoint], coeff_bits: int, start: int,
                   b: int = 0, p: int = 2) -> int:
    """
    Run one full tournament starting from a given coefficient.

    For each bit position (0 to coeff_bits-1):
    - Compare current coefficient with its partner (differs in that bit)
    - Keep the winner

    This is the core algorithm: O(k) comparisons for k-bit coefficients.

    Args:
        data: List of DataPoint objects
        coeff_bits: Number of bits in coefficient space
        start: Starting coefficient
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Final winning coefficient
    """
    current = start

    for bit in range(coeff_bits):
        partner = current ^ (1 << bit)
        current = tournament_winner(data, current, partner, b, p)

    return current


def find_optimal(data: List[DataPoint], coeff_bits: int,
                 start: Optional[int] = None, b: int = 0, p: int = 2) -> int:
    """
    Find the optimal coefficient using tournament optimization.

    For data where a unique optimum exists (perfect fit or clear winner),
    a single tournament from any starting point finds the optimal.

    Args:
        data: List of DataPoint objects
        coeff_bits: Number of bits in coefficient space
        start: Starting coefficient (default: 0)
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Optimal coefficient

    Example:
        >>> data = [DataPoint(1, 5), DataPoint(2, 10)]
        >>> find_optimal(data, coeff_bits=4)
        5
    """
    if start is None:
        start = 0
    return run_tournament(data, coeff_bits, start, b, p)


def find_optimal_monte_carlo(data: List[DataPoint], coeff_bits: int,
                              n_samples: int = 10, b: int = 0,
                              p: int = 2) -> Tuple[int, Dict[int, int]]:
    """
    Find optimal coefficient using Monte Carlo sampling.

    Runs multiple tournaments from random starting points and returns
    the most common winner. Useful when ties or multiple local optima exist.

    Args:
        data: List of DataPoint objects
        coeff_bits: Number of bits in coefficient space
        n_samples: Number of random starting points to try
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Tuple of (most_common_winner, {winner: count})

    Example:
        >>> data = [DataPoint(1, 5), DataPoint(2, 10)]
        >>> winner, counts = find_optimal_monte_carlo(data, 4, n_samples=10)
        >>> winner
        5
    """
    max_coeff = (1 << coeff_bits) - 1
    winner_counts = Counter()

    for _ in range(n_samples):
        start = random.randint(0, max_coeff)
        winner = run_tournament(data, coeff_bits, start, b, p)
        winner_counts[winner] += 1

    most_common = winner_counts.most_common(1)[0][0]
    return most_common, dict(winner_counts)


def brute_force_optimal(data: List[DataPoint], coeff_bits: int,
                        b: int = 0, p: int = 2) -> Tuple[int, int]:
    """
    Find optimal coefficient by exhaustive search.

    Checks all 2^coeff_bits possible coefficients. Use for verification
    or when coeff_bits is small.

    Args:
        data: List of DataPoint objects
        coeff_bits: Number of bits in coefficient space
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Tuple of (optimal_coefficient, max_valuation_sum)
    """
    best_coeff = 0
    best_sum = -float('inf')

    for coeff in range(1 << coeff_bits):
        val_sum = valuation_sum(data, coeff, b, p)
        if val_sum > best_sum:
            best_sum = val_sum
            best_coeff = coeff

    return best_coeff, best_sum


def analyze_tournament(data: List[DataPoint], coeff_bits: int,
                       b: int = 0, p: int = 2) -> Dict:
    """
    Analyze the tournament structure for debugging and understanding.

    Returns information about:
    - Valuation sums for all coefficients
    - Tournament outcomes from all starting points
    - Attractor basins (which starts lead to which winners)

    Args:
        data: List of DataPoint objects
        coeff_bits: Number of bits in coefficient space
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Dict with analysis results
    """
    n_coeffs = 1 << coeff_bits

    # Compute all valuation sums
    valuations = {}
    for coeff in range(n_coeffs):
        valuations[coeff] = valuation_sum(data, coeff, b, p)

    # Find all tournament outcomes
    outcomes = {}
    basins = {}
    for start in range(n_coeffs):
        winner = run_tournament(data, coeff_bits, start, b, p)
        outcomes[start] = winner
        if winner not in basins:
            basins[winner] = []
        basins[winner].append(start)

    # Find brute force optimal
    bf_optimal, bf_val = brute_force_optimal(data, coeff_bits, b, p)

    return {
        'valuations': valuations,
        'outcomes': outcomes,
        'basins': basins,
        'attractors': list(basins.keys()),
        'brute_force_optimal': bf_optimal,
        'brute_force_valuation': bf_val,
        'n_attractors': len(basins),
    }
