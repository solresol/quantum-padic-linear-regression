"""
Multi-Dimensional p-adic Regression

Extends tournament-based optimization to multiple coefficients:
y = a₁x₁ + a₂x₂ + ... + aₙxₙ + b

Approaches explored:
1. Joint tournament: Treat all coefficients as single bit-vector
2. Coordinate descent: Optimize each coefficient independently
3. Alternating optimization: Cycle through coefficients
"""

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from .core import padic_valuation, INFINITY_VALUATION


@dataclass
class MultiDimDataPoint:
    """A data point with multiple x features and one y target."""
    x: List[int]  # Feature vector
    y: int        # Target value

    def __post_init__(self):
        self.x = list(self.x)  # Ensure it's a list

    @property
    def dim(self) -> int:
        return len(self.x)


def multidim_valuation_sum(data: List[MultiDimDataPoint],
                           coeffs: List[int],
                           intercept: int = 0,
                           p: int = 2) -> int:
    """
    Compute valuation sum for multi-dimensional fit.

    For model y = sum(a_i * x_i) + b, compute sum of v_p(residuals).

    Args:
        data: List of MultiDimDataPoint objects
        coeffs: Coefficient vector [a₁, a₂, ..., aₙ]
        intercept: Intercept term b (default 0)
        p: Prime base (default 2)

    Returns:
        Total valuation sum (higher is better)
    """
    total = 0
    for point in data:
        prediction = sum(c * x for c, x in zip(coeffs, point.x)) + intercept
        residual = point.y - prediction
        total += padic_valuation(residual, p)
    return total


def brute_force_multidim(data: List[MultiDimDataPoint],
                         coeff_bits: int,
                         n_coeffs: int,
                         include_intercept: bool = False,
                         p: int = 2) -> Tuple[List[int], int, int]:
    """
    Brute force search over all coefficient combinations.

    WARNING: Complexity is O(2^(k*n)) where k=bits and n=coefficients.
    Only use for small problems!

    Args:
        data: List of MultiDimDataPoint objects
        coeff_bits: Bits per coefficient
        n_coeffs: Number of coefficients
        include_intercept: Whether to also search for intercept
        p: Prime base (default 2)

    Returns:
        Tuple of (best_coeffs, best_intercept, best_valuation)
    """
    max_coeff = 1 << coeff_bits
    best_coeffs = [0] * n_coeffs
    best_intercept = 0
    best_val = -float('inf')

    # Generate all coefficient combinations
    def gen_coeffs(remaining: int, current: List[int]):
        nonlocal best_coeffs, best_intercept, best_val

        if remaining == 0:
            # Evaluate this coefficient combination
            intercept_range = range(max_coeff) if include_intercept else [0]
            for b in intercept_range:
                val = multidim_valuation_sum(data, current, b, p)
                if val > best_val:
                    best_val = val
                    best_coeffs = current.copy()
                    best_intercept = b
            return

        for c in range(max_coeff):
            current.append(c)
            gen_coeffs(remaining - 1, current)
            current.pop()

    gen_coeffs(n_coeffs, [])
    return best_coeffs, best_intercept, best_val


def tournament_winner_multidim(data: List[MultiDimDataPoint],
                               coeffs1: List[int], coeffs2: List[int],
                               intercept: int = 0, p: int = 2) -> List[int]:
    """
    Compare two coefficient vectors and return the winner.
    """
    val1 = multidim_valuation_sum(data, coeffs1, intercept, p)
    val2 = multidim_valuation_sum(data, coeffs2, intercept, p)

    if val1 >= val2:
        return coeffs1.copy()
    return coeffs2.copy()


def coordinate_descent_tournament(data: List[MultiDimDataPoint],
                                   coeff_bits: int,
                                   n_coeffs: int,
                                   intercept: int = 0,
                                   max_iterations: int = 10,
                                   p: int = 2) -> Tuple[List[int], int]:
    """
    Optimize coefficients using coordinate descent with tournaments.

    For each coefficient in turn:
    - Hold others fixed
    - Run tournament to find optimal value for this coefficient

    Args:
        data: List of MultiDimDataPoint objects
        coeff_bits: Bits per coefficient
        n_coeffs: Number of coefficients
        intercept: Fixed intercept (default 0)
        max_iterations: Maximum passes through all coefficients
        p: Prime base (default 2)

    Returns:
        Tuple of (optimal_coeffs, final_valuation)
    """
    # Initialize with zeros or random
    coeffs = [0] * n_coeffs

    for iteration in range(max_iterations):
        changed = False

        for i in range(n_coeffs):
            # Optimize coefficient i, holding others fixed
            current = coeffs[i]

            for bit in range(coeff_bits):
                # Compare current vs flipped bit
                partner = current ^ (1 << bit)

                # Create coefficient vectors for comparison
                coeffs1 = coeffs.copy()
                coeffs2 = coeffs.copy()
                coeffs1[i] = current
                coeffs2[i] = partner

                winner = tournament_winner_multidim(data, coeffs1, coeffs2,
                                                    intercept, p)
                new_val = winner[i]

                if new_val != current:
                    changed = True
                    current = new_val

            coeffs[i] = current

        if not changed:
            break

    final_val = multidim_valuation_sum(data, coeffs, intercept, p)
    return coeffs, final_val


def joint_tournament_multidim(data: List[MultiDimDataPoint],
                               coeff_bits: int,
                               n_coeffs: int,
                               intercept: int = 0,
                               start: Optional[List[int]] = None,
                               p: int = 2) -> Tuple[List[int], int]:
    """
    Joint tournament: treat all coefficients as one long bit-vector.

    Total bits = coeff_bits * n_coeffs
    Each round flips one bit in the combined vector.

    Args:
        data: List of MultiDimDataPoint objects
        coeff_bits: Bits per coefficient
        n_coeffs: Number of coefficients
        intercept: Fixed intercept (default 0)
        start: Starting coefficient vector (default: all zeros)
        p: Prime base (default 2)

    Returns:
        Tuple of (optimal_coeffs, final_valuation)
    """
    if start is None:
        start = [0] * n_coeffs

    coeffs = start.copy()
    total_bits = coeff_bits * n_coeffs

    for global_bit in range(total_bits):
        # Determine which coefficient and which bit
        coeff_idx = global_bit // coeff_bits
        local_bit = global_bit % coeff_bits

        # Create partner by flipping this bit
        partner_coeffs = coeffs.copy()
        partner_coeffs[coeff_idx] ^= (1 << local_bit)

        # Compare and keep winner
        coeffs = tournament_winner_multidim(data, coeffs, partner_coeffs,
                                            intercept, p)

    final_val = multidim_valuation_sum(data, coeffs, intercept, p)
    return coeffs, final_val


def monte_carlo_multidim(data: List[MultiDimDataPoint],
                          coeff_bits: int,
                          n_coeffs: int,
                          n_samples: int = 10,
                          method: str = 'joint',
                          intercept: int = 0,
                          p: int = 2) -> Tuple[List[int], Dict]:
    """
    Monte Carlo multi-dimensional optimization.

    Runs multiple tournaments from random starting points.

    Args:
        data: List of MultiDimDataPoint objects
        coeff_bits: Bits per coefficient
        n_coeffs: Number of coefficients
        n_samples: Number of random starting points
        method: 'joint' or 'coordinate'
        intercept: Fixed intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Tuple of (most_common_winner, winner_counts)
    """
    max_coeff = (1 << coeff_bits) - 1
    winner_counts = Counter()

    for _ in range(n_samples):
        start = [random.randint(0, max_coeff) for _ in range(n_coeffs)]

        if method == 'joint':
            winner, _ = joint_tournament_multidim(data, coeff_bits, n_coeffs,
                                                   intercept, start, p)
        else:  # coordinate
            winner, _ = coordinate_descent_tournament(data, coeff_bits, n_coeffs,
                                                       intercept, 1, p)

        winner_tuple = tuple(winner)
        winner_counts[winner_tuple] += 1

    most_common = list(winner_counts.most_common(1)[0][0])
    return most_common, dict(winner_counts)
