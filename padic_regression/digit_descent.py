"""
Digit-by-Digit p-adic Regression

A new approach that's O(p^n) per digit instead of O(p^(k*n)) brute force,
where p is the prime, n is the number of coefficients, and k is digits.

Algorithm:
1. Randomly pick coefficients, express as p-adic integers
2. For each digit position:
   - Create hypercube of all p^n variations for that digit
   - Evaluate loss on entire hypercube
   - Track which digit values appear in best solutions
3. Repeat with multiple random starting points
4. Use statistical analysis to identify likely optimal digits
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math

from .core import padic_valuation, INFINITY_VALUATION
from .multidim import MultiDimDataPoint, multidim_valuation_sum


def int_to_padic_digits(n: int, p: int, num_digits: int) -> List[int]:
    """
    Convert integer to p-adic digit representation.

    Returns list of digits [d_0, d_1, ...] where n = d_0 + d_1*p + d_2*p^2 + ...
    Least significant digit first.
    """
    if n < 0:
        raise ValueError("Negative numbers not yet supported")

    digits = []
    for _ in range(num_digits):
        digits.append(n % p)
        n //= p
    return digits


def padic_digits_to_int(digits: List[int], p: int) -> int:
    """Convert p-adic digits back to integer."""
    result = 0
    power = 1
    for d in digits:
        result += d * power
        power *= p
    return result


def generate_hypercube(base_coeffs: List[List[int]], digit_pos: int,
                       p: int) -> List[List[List[int]]]:
    """
    Generate all p^n variations for a single digit position.

    Args:
        base_coeffs: List of coefficient digit representations
        digit_pos: Which digit position to vary
        p: Prime base

    Returns:
        List of all p^n coefficient variations
    """
    n_coeffs = len(base_coeffs)
    variations = []

    # Generate all p^n combinations for digit at digit_pos
    def generate(coeff_idx: int, current: List[List[int]]):
        if coeff_idx == n_coeffs:
            variations.append([c.copy() for c in current])
            return

        for digit_val in range(p):
            current[coeff_idx][digit_pos] = digit_val
            generate(coeff_idx + 1, current)

    # Start with copy of base
    current = [c.copy() for c in base_coeffs]
    generate(0, current)

    return variations


@dataclass
class DigitStatistics:
    """Track statistics for digit values across samples."""
    counts: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    best_vals: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))

    def record(self, coeff_idx: int, digit_val: int, valuation: int):
        """Record a digit value and its associated valuation."""
        self.counts[coeff_idx][digit_val] += 1
        self.best_vals[coeff_idx].append((valuation, digit_val))

    def get_best_digit(self, coeff_idx: int, top_k: int = 5) -> Tuple[int, float]:
        """
        Get the most likely optimal digit for a coefficient.

        Returns (best_digit, confidence) where confidence is how much
        more often it appears in top results vs random chance.
        """
        if coeff_idx not in self.best_vals:
            return 0, 0.0

        # Sort by valuation (higher is better) and take top k
        sorted_results = sorted(self.best_vals[coeff_idx], reverse=True)[:top_k]

        if not sorted_results:
            return 0, 0.0

        # Count digit occurrences in top results
        top_counts = Counter(d for _, d in sorted_results)
        best_digit = top_counts.most_common(1)[0][0]

        # Confidence: how much more than random (1/p)?
        p = max(self.counts[coeff_idx].keys()) + 1 if self.counts[coeff_idx] else 2
        random_prob = 1.0 / p
        observed_prob = top_counts[best_digit] / len(sorted_results)
        confidence = observed_prob / random_prob

        return best_digit, confidence


def aggregate_digit_valuations(results: List[Tuple[int, List[int], int]],
                               n_coeffs: int,
                               p: int) -> Dict[int, Dict[int, float]]:
    """
    Aggregate valuations by digit value for each coefficient.

    Returns {coeff_idx: {digit_val: total_valuation}}
    """
    aggregated = defaultdict(lambda: defaultdict(float))

    for valuation, digits, int_digit in results:
        for coeff_idx, digit_val in enumerate(digits):
            aggregated[coeff_idx][digit_val] += valuation
        # Intercept
        aggregated[n_coeffs][int_digit] += valuation

    return aggregated


def explore_digit_hypercube(data: List[MultiDimDataPoint],
                            base_coeffs_digits: List[List[int]],
                            digit_pos: int,
                            p: int,
                            intercept_digits: Optional[List[int]] = None
                            ) -> List[Tuple[int, List[int], int]]:
    """
    Explore all p^n combinations for one digit position.

    Args:
        data: Data points
        base_coeffs_digits: Coefficient digit representations
        digit_pos: Which digit to vary
        p: Prime base
        intercept_digits: Intercept digit representation (optional)

    Returns:
        List of (valuation, digit_values, intercept_digit) for each hypercube point
    """
    n_coeffs = len(base_coeffs_digits)
    include_intercept = intercept_digits is not None

    results = []

    # Generate all combinations for this digit
    def explore(coeff_idx: int, current_digits: List[int],
                current_coeffs: List[List[int]]):
        if coeff_idx == n_coeffs:
            # If we have intercept, explore its digit too
            if include_intercept:
                for b_digit in range(p):
                    int_digits = intercept_digits.copy()
                    int_digits[digit_pos] = b_digit

                    # Convert to integers and evaluate
                    coeffs = [padic_digits_to_int(c, p) for c in current_coeffs]
                    intercept = padic_digits_to_int(int_digits, p)

                    val = multidim_valuation_sum(data, coeffs, intercept, p)
                    results.append((val, current_digits.copy(), b_digit))
            else:
                coeffs = [padic_digits_to_int(c, p) for c in current_coeffs]
                val = multidim_valuation_sum(data, coeffs, 0, p)
                results.append((val, current_digits.copy(), 0))
            return

        for digit_val in range(p):
            current_coeffs[coeff_idx][digit_pos] = digit_val
            current_digits.append(digit_val)
            explore(coeff_idx + 1, current_digits, current_coeffs)
            current_digits.pop()

    current_coeffs = [c.copy() for c in base_coeffs_digits]
    explore(0, [], current_coeffs)

    return results


def digit_descent_regression(data: List[MultiDimDataPoint],
                              n_coeffs: int,
                              num_digits: int,
                              p: int = 2,
                              n_samples: int = 10,
                              include_intercept: bool = False,
                              confidence_threshold: float = 1.5,
                              verbose: bool = False
                              ) -> Tuple[List[int], int, Dict]:
    """
    Find optimal coefficients using digit-by-digit descent.

    Complexity: O(num_digits * n_samples * p^n_coeffs)
    Compare to brute force: O(p^(num_digits * n_coeffs))

    Args:
        data: List of MultiDimDataPoint
        n_coeffs: Number of slope coefficients
        num_digits: Number of p-adic digits to use
        p: Prime base (default 2)
        n_samples: Random starting points per digit
        include_intercept: Whether to include intercept term
        confidence_threshold: Min confidence to accept a digit
        verbose: Print progress

    Returns:
        Tuple of (coefficients, intercept, statistics)
    """
    max_coeff = p ** num_digits - 1
    total_coeffs = n_coeffs + (1 if include_intercept else 0)

    # Initialize with zeros
    best_coeffs_digits = [[0] * num_digits for _ in range(n_coeffs)]
    best_intercept_digits = [0] * num_digits if include_intercept else None

    stats = {
        'digit_stats': [],
        'evaluations': 0,
        'confident_digits': 0,
    }

    # Process each digit position (least significant first)
    for digit_pos in range(num_digits):
        if verbose:
            print(f"\n--- Digit position {digit_pos} ---")

        digit_stats = DigitStatistics()

        for sample in range(n_samples):
            # Random perturbation for higher digits (not yet determined)
            sample_coeffs = [c.copy() for c in best_coeffs_digits]
            for c in sample_coeffs:
                for d in range(digit_pos + 1, num_digits):
                    c[d] = random.randint(0, p - 1)

            if include_intercept:
                sample_intercept = best_intercept_digits.copy()
                for d in range(digit_pos + 1, num_digits):
                    sample_intercept[d] = random.randint(0, p - 1)
            else:
                sample_intercept = None

            # Explore hypercube for this digit
            results = explore_digit_hypercube(
                data, sample_coeffs, digit_pos, p, sample_intercept
            )
            stats['evaluations'] += len(results)

            # Find best result in this hypercube
            best_val, best_digits, best_int_digit = max(results)

            # Record statistics
            for coeff_idx, digit_val in enumerate(best_digits):
                digit_stats.record(coeff_idx, digit_val, best_val)
            if include_intercept:
                digit_stats.record(n_coeffs, best_int_digit, best_val)

        # Determine best digits from statistics
        for coeff_idx in range(n_coeffs):
            best_digit, confidence = digit_stats.get_best_digit(coeff_idx)
            best_coeffs_digits[coeff_idx][digit_pos] = best_digit
            if confidence >= confidence_threshold:
                stats['confident_digits'] += 1
            if verbose:
                print(f"  Coeff {coeff_idx}: digit={best_digit}, confidence={confidence:.2f}")

        if include_intercept:
            best_digit, confidence = digit_stats.get_best_digit(n_coeffs)
            best_intercept_digits[digit_pos] = best_digit
            if confidence >= confidence_threshold:
                stats['confident_digits'] += 1
            if verbose:
                print(f"  Intercept: digit={best_digit}, confidence={confidence:.2f}")

        stats['digit_stats'].append(digit_stats)

    # Convert to integers
    final_coeffs = [padic_digits_to_int(c, p) for c in best_coeffs_digits]
    final_intercept = padic_digits_to_int(best_intercept_digits, p) if include_intercept else 0

    return final_coeffs, final_intercept, stats


def digit_descent_v2(data: List[MultiDimDataPoint],
                      n_coeffs: int,
                      num_digits: int,
                      p: int = 2,
                      n_samples: int = 10,
                      include_intercept: bool = False,
                      verbose: bool = False
                      ) -> Tuple[List[int], int, Dict]:
    """
    Improved digit descent using aggregate valuations.

    Instead of picking the digit that appears in "best" results,
    we sum all valuations for each digit value and pick the highest.

    This is more robust because it considers all hypercube points,
    not just the winner.
    """
    max_coeff = p ** num_digits - 1
    total_coeffs = n_coeffs + (1 if include_intercept else 0)

    # Initialize with zeros
    best_coeffs_digits = [[0] * num_digits for _ in range(n_coeffs)]
    best_intercept_digits = [0] * num_digits if include_intercept else None

    stats = {
        'evaluations': 0,
        'digit_choices': [],
    }

    # Process each digit position
    for digit_pos in range(num_digits):
        if verbose:
            print(f"\n--- Digit position {digit_pos} ---")

        # Accumulate valuations across all samples
        total_valuations = defaultdict(lambda: defaultdict(float))
        total_counts = defaultdict(lambda: defaultdict(int))

        for sample in range(n_samples):
            # Random perturbation for higher digits
            sample_coeffs = [c.copy() for c in best_coeffs_digits]
            for c in sample_coeffs:
                for d in range(digit_pos + 1, num_digits):
                    c[d] = random.randint(0, p - 1)

            if include_intercept:
                sample_intercept = best_intercept_digits.copy()
                for d in range(digit_pos + 1, num_digits):
                    sample_intercept[d] = random.randint(0, p - 1)
            else:
                sample_intercept = None

            # Explore hypercube
            results = explore_digit_hypercube(
                data, sample_coeffs, digit_pos, p, sample_intercept
            )
            stats['evaluations'] += len(results)

            # Aggregate valuations
            for valuation, digits, int_digit in results:
                for coeff_idx, digit_val in enumerate(digits):
                    total_valuations[coeff_idx][digit_val] += valuation
                    total_counts[coeff_idx][digit_val] += 1
                if include_intercept:
                    total_valuations[n_coeffs][int_digit] += valuation
                    total_counts[n_coeffs][int_digit] += 1

        # Pick best digit for each coefficient based on total valuation
        digit_choices = []
        for coeff_idx in range(n_coeffs):
            best_digit = max(range(p),
                           key=lambda d: total_valuations[coeff_idx][d])
            avg_val = total_valuations[coeff_idx][best_digit] / max(1, total_counts[coeff_idx][best_digit])
            best_coeffs_digits[coeff_idx][digit_pos] = best_digit
            digit_choices.append((coeff_idx, best_digit, avg_val))
            if verbose:
                print(f"  Coeff {coeff_idx}: digit={best_digit} (avg_val={avg_val:.1f})")

        if include_intercept:
            best_digit = max(range(p),
                           key=lambda d: total_valuations[n_coeffs][d])
            best_intercept_digits[digit_pos] = best_digit
            if verbose:
                print(f"  Intercept: digit={best_digit}")

        stats['digit_choices'].append(digit_choices)

    # Convert to integers
    final_coeffs = [padic_digits_to_int(c, p) for c in best_coeffs_digits]
    final_intercept = padic_digits_to_int(best_intercept_digits, p) if include_intercept else 0

    return final_coeffs, final_intercept, stats


def digit_descent_msb_first(data: List[MultiDimDataPoint],
                             n_coeffs: int,
                             num_digits: int,
                             p: int = 2,
                             n_samples: int = 10,
                             include_intercept: bool = False,
                             verbose: bool = False
                             ) -> Tuple[List[int], int, Dict]:
    """
    Digit descent processing MOST significant digit first.

    The intuition: higher-order digits have more impact on the value,
    so we should lock them in first. Lower digits can then be optimized
    in the context of correct higher digits.
    """
    max_coeff = p ** num_digits - 1

    # Initialize with zeros
    best_coeffs_digits = [[0] * num_digits for _ in range(n_coeffs)]
    best_intercept_digits = [0] * num_digits if include_intercept else None

    stats = {
        'evaluations': 0,
        'digit_choices': [],
    }

    # Process from most significant to least significant
    for digit_pos in range(num_digits - 1, -1, -1):
        if verbose:
            print(f"\n--- Digit position {digit_pos} (MSB first) ---")

        digit_stats = DigitStatistics()

        for sample in range(n_samples):
            # Random perturbation for LOWER digits (not yet determined)
            sample_coeffs = [c.copy() for c in best_coeffs_digits]
            for c in sample_coeffs:
                for d in range(0, digit_pos):
                    c[d] = random.randint(0, p - 1)

            if include_intercept:
                sample_intercept = best_intercept_digits.copy()
                for d in range(0, digit_pos):
                    sample_intercept[d] = random.randint(0, p - 1)
            else:
                sample_intercept = None

            # Explore hypercube for this digit
            results = explore_digit_hypercube(
                data, sample_coeffs, digit_pos, p, sample_intercept
            )
            stats['evaluations'] += len(results)

            # Find best result
            best_val, best_digits, best_int_digit = max(results)

            for coeff_idx, digit_val in enumerate(best_digits):
                digit_stats.record(coeff_idx, digit_val, best_val)
            if include_intercept:
                digit_stats.record(n_coeffs, best_int_digit, best_val)

        # Determine best digits
        for coeff_idx in range(n_coeffs):
            best_digit, confidence = digit_stats.get_best_digit(coeff_idx)
            best_coeffs_digits[coeff_idx][digit_pos] = best_digit
            if verbose:
                print(f"  Coeff {coeff_idx}: digit={best_digit}, conf={confidence:.2f}")

        if include_intercept:
            best_digit, confidence = digit_stats.get_best_digit(n_coeffs)
            best_intercept_digits[digit_pos] = best_digit
            if verbose:
                print(f"  Intercept: digit={best_digit}")

    # Convert to integers
    final_coeffs = [padic_digits_to_int(c, p) for c in best_coeffs_digits]
    final_intercept = padic_digits_to_int(best_intercept_digits, p) if include_intercept else 0

    return final_coeffs, final_intercept, stats


def digit_descent_with_refinement(data: List[MultiDimDataPoint],
                                   n_coeffs: int,
                                   num_digits: int,
                                   p: int = 2,
                                   n_samples: int = 10,
                                   n_refinement_passes: int = 2,
                                   include_intercept: bool = False,
                                   verbose: bool = False
                                   ) -> Tuple[List[int], int, Dict]:
    """
    Digit descent with multiple refinement passes.

    After initial pass, re-run with the discovered digits as base,
    which can correct early mistakes.
    """
    coeffs, intercept, stats = digit_descent_regression(
        data, n_coeffs, num_digits, p, n_samples, include_intercept,
        verbose=verbose
    )

    for pass_num in range(n_refinement_passes):
        if verbose:
            print(f"\n=== Refinement pass {pass_num + 1} ===")

        # Use current best as starting point
        prev_coeffs = coeffs
        prev_intercept = intercept

        coeffs, intercept, new_stats = digit_descent_regression(
            data, n_coeffs, num_digits, p, n_samples, include_intercept,
            verbose=verbose
        )

        stats['evaluations'] += new_stats['evaluations']

        # Check if we've converged
        if coeffs == prev_coeffs and intercept == prev_intercept:
            if verbose:
                print("Converged!")
            break

    return coeffs, intercept, stats
