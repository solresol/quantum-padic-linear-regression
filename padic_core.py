#!/usr/bin/env python3
"""
Core p-adic Functions

Shared functions and classes used by both padic_regression.py and quantum_oracle.py.
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class DataPoint:
    """A data point with x and y coordinates."""
    x: int
    y: int


def classical_2adic_valuation(n: int) -> int:
    """
    Compute the 2-adic valuation of n (number of trailing zeros in binary).
    Returns infinity (represented as large int) for n=0.
    """
    if n == 0:
        return 1000  # Represents infinity
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    return count


def classical_2adic_distance(a: int, b: int) -> float:
    """Compute the 2-adic distance between a and b."""
    if a == b:
        return 0.0
    v = classical_2adic_valuation(abs(a - b))
    return 2.0 ** (-v)


def classical_residual_sum(data: List[DataPoint], m: int, b: int) -> float:
    """
    Compute the sum of 2-adic distances for a line y = mx + b.
    """
    total = 0.0
    for point in data:
        residual = point.y - (m * point.x + b)
        total += classical_2adic_distance(residual, 0)
    return total


def classical_valuation_sum(data: List[DataPoint], m: int, b: int) -> int:
    """
    Compute the sum of 2-adic valuations for a line y = mx + b.
    Higher is better (means smaller total distance).
    """
    total = 0
    for point in data:
        residual = point.y - (m * point.x + b)
        total += classical_2adic_valuation(residual)
    return total


def bits_needed(max_val: int) -> int:
    """Number of bits needed to represent values up to max_val."""
    if max_val <= 0:
        return 1
    return max(1, math.ceil(math.log2(max_val + 1)))
