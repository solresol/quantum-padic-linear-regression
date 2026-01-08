#!/usr/bin/env python3
"""
Multi-Dimensional p-adic Regression Experiments

Tests the tournament-based optimization for multiple coefficients:
y = a₁x₁ + a₂x₂ + ... + aₙxₙ

Key questions:
1. Does joint tournament (all bits at once) find the optimal?
2. Does coordinate descent (one coefficient at a time) work?
3. How does accuracy scale with dimensions?
4. What is the speedup vs brute force?
"""

import random
import time
from typing import List, Tuple

from padic_regression import (
    MultiDimDataPoint,
    multidim_valuation_sum,
    brute_force_multidim,
    coordinate_descent_tournament,
    joint_tournament_multidim,
    monte_carlo_multidim,
)


def generate_perfect_data(true_coeffs: List[int], n_points: int,
                          x_range: int = 4) -> List[MultiDimDataPoint]:
    """Generate data that perfectly fits y = sum(a_i * x_i)."""
    data = []
    n_coeffs = len(true_coeffs)

    for _ in range(n_points):
        x = [random.randint(1, x_range) for _ in range(n_coeffs)]
        y = sum(c * xi for c, xi in zip(true_coeffs, x))
        data.append(MultiDimDataPoint(x, y))

    return data


def generate_noisy_data(true_coeffs: List[int], n_points: int,
                        x_range: int = 4, noise_bits: int = 1) -> List[MultiDimDataPoint]:
    """Generate data with small noise: y = sum(a_i * x_i) + noise."""
    data = []
    n_coeffs = len(true_coeffs)

    for _ in range(n_points):
        x = [random.randint(1, x_range) for _ in range(n_coeffs)]
        noise = random.randint(-(1 << noise_bits), (1 << noise_bits))
        y = sum(c * xi for c, xi in zip(true_coeffs, x)) + noise
        data.append(MultiDimDataPoint(x, y))

    return data


def experiment_2d_perfect():
    """Test 2-dimensional regression with perfect data."""
    print("=" * 60)
    print("Experiment: 2D Perfect Data")
    print("y = a₁x₁ + a₂x₂")
    print("=" * 60)

    true_coeffs = [3, 5]
    coeff_bits = 3
    n_points = 3

    data = generate_perfect_data(true_coeffs, n_points)
    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Data points: {[(pt.x, pt.y) for pt in data]}")

    # Brute force (for verification)
    print("\n--- Brute Force ---")
    start = time.time()
    bf_coeffs, bf_intercept, bf_val = brute_force_multidim(data, coeff_bits, 2)
    bf_time = time.time() - start
    print(f"Result: {bf_coeffs}, val={bf_val}, time={bf_time:.4f}s")
    print(f"Correct: {bf_coeffs == true_coeffs}")

    # Joint tournament
    print("\n--- Joint Tournament ---")
    start = time.time()
    jt_coeffs, jt_val = joint_tournament_multidim(data, coeff_bits, 2)
    jt_time = time.time() - start
    print(f"Result: {jt_coeffs}, val={jt_val}, time={jt_time:.4f}s")
    print(f"Correct: {jt_coeffs == true_coeffs}")

    # Coordinate descent
    print("\n--- Coordinate Descent ---")
    start = time.time()
    cd_coeffs, cd_val = coordinate_descent_tournament(data, coeff_bits, 2)
    cd_time = time.time() - start
    print(f"Result: {cd_coeffs}, val={cd_val}, time={cd_time:.4f}s")
    print(f"Correct: {cd_coeffs == true_coeffs}")


def experiment_accuracy_vs_dimensions():
    """Test how accuracy changes with number of dimensions."""
    print("\n" + "=" * 60)
    print("Experiment: Accuracy vs Dimensions")
    print("=" * 60)

    coeff_bits = 3
    n_points = 5
    n_trials = 20

    for n_dims in [1, 2, 3, 4]:
        print(f"\n--- {n_dims} dimensions ---")

        joint_correct = 0
        coord_correct = 0

        for trial in range(n_trials):
            # Random true coefficients
            true_coeffs = [random.randint(1, (1 << coeff_bits) - 1)
                          for _ in range(n_dims)]
            data = generate_perfect_data(true_coeffs, n_points)

            # Brute force (ground truth)
            bf_coeffs, _, _ = brute_force_multidim(data, coeff_bits, n_dims)

            # Joint tournament
            jt_coeffs, _ = joint_tournament_multidim(data, coeff_bits, n_dims)
            if jt_coeffs == bf_coeffs:
                joint_correct += 1

            # Coordinate descent
            cd_coeffs, _ = coordinate_descent_tournament(data, coeff_bits, n_dims)
            if cd_coeffs == bf_coeffs:
                coord_correct += 1

        print(f"  Joint tournament: {joint_correct}/{n_trials} ({100*joint_correct/n_trials:.0f}%)")
        print(f"  Coordinate descent: {coord_correct}/{n_trials} ({100*coord_correct/n_trials:.0f}%)")


def experiment_speedup():
    """Compare runtime of tournament vs brute force."""
    print("\n" + "=" * 60)
    print("Experiment: Speedup Analysis")
    print("=" * 60)

    n_points = 3
    n_dims = 2

    print(f"\n{'Bits':>4} | {'Search Space':>12} | {'Brute Force':>12} | {'Tournament':>12} | {'Speedup':>10}")
    print("-" * 60)

    for coeff_bits in [2, 3, 4, 5]:
        true_coeffs = [random.randint(1, (1 << coeff_bits) - 1)
                      for _ in range(n_dims)]
        data = generate_perfect_data(true_coeffs, n_points)

        search_space = (1 << coeff_bits) ** n_dims

        # Brute force timing
        start = time.time()
        for _ in range(5):
            brute_force_multidim(data, coeff_bits, n_dims)
        bf_time = (time.time() - start) / 5

        # Tournament timing
        start = time.time()
        for _ in range(5):
            joint_tournament_multidim(data, coeff_bits, n_dims)
        jt_time = (time.time() - start) / 5

        speedup = bf_time / jt_time if jt_time > 0 else float('inf')

        print(f"{coeff_bits:>4} | {search_space:>12,} | {bf_time:>10.4f}s | {jt_time:>10.4f}s | {speedup:>10.1f}x")


def experiment_monte_carlo():
    """Test Monte Carlo approach for noisy data."""
    print("\n" + "=" * 60)
    print("Experiment: Monte Carlo for Noisy Data")
    print("=" * 60)

    true_coeffs = [3, 5]
    coeff_bits = 3
    n_points = 4

    # Generate noisy data
    data = generate_noisy_data(true_coeffs, n_points, noise_bits=1)
    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Noisy data: {[(pt.x, pt.y) for pt in data]}")

    # Brute force
    bf_coeffs, _, bf_val = brute_force_multidim(data, coeff_bits, 2)
    print(f"\nBrute force optimal: {bf_coeffs}, val={bf_val}")

    # Monte Carlo joint
    print("\n--- Monte Carlo (Joint) ---")
    for n_samples in [1, 5, 10, 50]:
        mc_coeffs, counts = monte_carlo_multidim(data, coeff_bits, 2,
                                                  n_samples, method='joint')
        correct = mc_coeffs == list(bf_coeffs)
        print(f"  {n_samples:2d} samples: {mc_coeffs}, correct={correct}")

    # Monte Carlo coordinate descent
    print("\n--- Monte Carlo (Coordinate Descent) ---")
    for n_samples in [1, 5, 10, 50]:
        mc_coeffs, counts = monte_carlo_multidim(data, coeff_bits, 2,
                                                  n_samples, method='coordinate')
        correct = mc_coeffs == list(bf_coeffs)
        print(f"  {n_samples:2d} samples: {mc_coeffs}, correct={correct}")


def experiment_convergence():
    """Analyze tournament convergence structure."""
    print("\n" + "=" * 60)
    print("Experiment: Convergence Analysis")
    print("=" * 60)

    true_coeffs = [2, 3]
    coeff_bits = 2
    n_points = 2
    data = generate_perfect_data(true_coeffs, n_points)

    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Data: {[(pt.x, pt.y) for pt in data]}")

    # Show all valuation sums
    print("\nValuation sums for all coefficient pairs:")
    max_c = 1 << coeff_bits
    for a1 in range(max_c):
        for a2 in range(max_c):
            val = multidim_valuation_sum(data, [a1, a2])
            print(f"  [{a1}, {a2}]: val={val}")

    # Show which starting points lead to which winners
    print("\nTournament outcomes from all starting points:")
    winner_map = {}
    for a1 in range(max_c):
        for a2 in range(max_c):
            start = [a1, a2]
            winner, _ = joint_tournament_multidim(data, coeff_bits, 2, start=start)
            winner_tuple = tuple(winner)
            if winner_tuple not in winner_map:
                winner_map[winner_tuple] = []
            winner_map[winner_tuple].append((a1, a2))

    for winner, starts in sorted(winner_map.items()):
        print(f"  Winner {list(winner)}: from {len(starts)} starting points")


def run_all_experiments():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("MULTI-DIMENSIONAL P-ADIC REGRESSION EXPERIMENTS")
    print("=" * 70)

    experiment_2d_perfect()
    experiment_accuracy_vs_dimensions()
    experiment_speedup()
    experiment_monte_carlo()
    experiment_convergence()


if __name__ == "__main__":
    run_all_experiments()
