"""
Core p-adic Functions

Provides fundamental operations for p-adic number theory:
- p-adic valuation (counting factors of prime p)
- p-adic distance
- Valuation sums for regression
"""

import math
from dataclasses import dataclass
from typing import List, Union

# Large value representing "infinity" for valuation of zero
INFINITY_VALUATION = 1000


@dataclass
class DataPoint:
    """A data point with x and y coordinates."""
    x: int
    y: int

    def __iter__(self):
        """Allow unpacking: x, y = point"""
        return iter((self.x, self.y))


def padic_valuation(n: int, p: int = 2) -> int:
    """
    Compute the p-adic valuation of n.

    The p-adic valuation v_p(n) is the largest power of p that divides n.
    For p=2, this equals the number of trailing zeros in binary.

    Args:
        n: Integer to compute valuation for
        p: Prime base (default 2)

    Returns:
        v_p(n), or INFINITY_VALUATION if n=0

    Examples:
        >>> padic_valuation(12, 2)  # 12 = 4 * 3 = 2^2 * 3
        2
        >>> padic_valuation(18, 3)  # 18 = 9 * 2 = 3^2 * 2
        2
        >>> padic_valuation(7, 2)   # 7 is odd
        0
    """
    if n == 0:
        return INFINITY_VALUATION

    n = abs(n)
    count = 0
    while n % p == 0:
        count += 1
        n //= p
    return count


def padic_distance(a: int, b: int, p: int = 2) -> float:
    """
    Compute the p-adic distance between a and b.

    The p-adic distance is d_p(a, b) = p^(-v_p(a-b)).

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default 2)

    Returns:
        p-adic distance as float

    Examples:
        >>> padic_distance(0, 8, 2)  # 8 = 2^3, so distance = 2^(-3) = 0.125
        0.125
        >>> padic_distance(3, 3, 2)  # Same number, distance = 0
        0.0
    """
    if a == b:
        return 0.0
    v = padic_valuation(a - b, p)
    if v >= INFINITY_VALUATION:
        return 0.0
    return float(p) ** (-v)


def valuation_sum(data: List[DataPoint], m: int, b: int = 0, p: int = 2) -> int:
    """
    Compute the sum of p-adic valuations for a linear fit.

    For line y = mx + b, computes sum of v_p(y_i - m*x_i - b) for all data points.
    Higher valuation sum means better fit in the p-adic sense.

    Args:
        data: List of DataPoint objects
        m: Slope coefficient
        b: Intercept (default 0 for line through origin)
        p: Prime base (default 2)

    Returns:
        Total valuation sum (higher is better)

    Examples:
        >>> data = [DataPoint(1, 2), DataPoint(2, 4)]
        >>> valuation_sum(data, 2, 0)  # Perfect fit: residuals are 0
        2000  # Returns INFINITY_VALUATION * len(data)
        >>> valuation_sum(data, 3, 0)  # Imperfect fit
        2  # Some finite valuation
    """
    total = 0
    for point in data:
        residual = point.y - (m * point.x + b)
        total += padic_valuation(residual, p)
    return total


def distance_sum(data: List[DataPoint], m: int, b: int = 0, p: int = 2) -> float:
    """
    Compute the sum of p-adic distances for a linear fit.

    For line y = mx + b, computes sum of d_p(y_i, m*x_i + b) for all data points.
    Lower distance sum means better fit.

    Args:
        data: List of DataPoint objects
        m: Slope coefficient
        b: Intercept (default 0)
        p: Prime base (default 2)

    Returns:
        Total distance sum (lower is better)
    """
    total = 0.0
    for point in data:
        residual = point.y - (m * point.x + b)
        total += padic_distance(residual, 0, p)
    return total


def bits_needed(max_val: int) -> int:
    """
    Number of bits needed to represent values up to max_val.

    Args:
        max_val: Maximum value to represent

    Returns:
        Minimum number of bits required

    Examples:
        >>> bits_needed(7)
        3
        >>> bits_needed(8)
        4
        >>> bits_needed(255)
        8
    """
    if max_val <= 0:
        return 1
    return max(1, math.ceil(math.log2(max_val + 1)))
