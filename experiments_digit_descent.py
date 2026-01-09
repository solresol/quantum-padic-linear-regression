#!/usr/bin/env python3
"""
Digit Descent Algorithm Experiments

Tests the new digit-by-digit p-adic regression approach.

Complexity comparison:
- Brute force: O(p^(k*n)) where k=digits, n=coefficients
- Digit descent: O(k * samples * p^n)

For p=2, k=8, n=3:
- Brute force: 2^24 = 16,777,216 evaluations
- Digit descent: 8 * 10 * 8 = 640 evaluations (with 10 samples)
"""

import random
import time
from typing import List

from padic_regression import (
    MultiDimDataPoint,
    multidim_valuation_sum,
    brute_force_multidim,
    digit_descent_regression,
    digit_descent_with_refinement,
    int_to_padic_digits,
    padic_digits_to_int,
)


def generate_perfect_data(true_coeffs: List[int], n_points: int,
                          x_range: int = 4, intercept: int = 0) -> List[MultiDimDataPoint]:
    """Generate data that perfectly fits y = sum(a_i * x_i) + b."""
    data = []
    n_coeffs = len(true_coeffs)

    for _ in range(n_points):
        x = [random.randint(1, x_range) for _ in range(n_coeffs)]
        y = sum(c * xi for c, xi in zip(true_coeffs, x)) + intercept
        data.append(MultiDimDataPoint(x, y))

    return data


def generate_noisy_data(true_coeffs: List[int], n_points: int,
                        x_range: int = 4, noise_bits: int = 1,
                        intercept: int = 0) -> List[MultiDimDataPoint]:
    """Generate noisy data."""
    data = []
    n_coeffs = len(true_coeffs)

    for _ in range(n_points):
        x = [random.randint(1, x_range) for _ in range(n_coeffs)]
        noise = random.randint(-(1 << noise_bits), (1 << noise_bits))
        y = sum(c * xi for c, xi in zip(true_coeffs, x)) + intercept + noise
        data.append(MultiDimDataPoint(x, y))

    return data


def experiment_basic_2d():
    """Basic test: 2D regression with perfect data."""
    print("=" * 60)
    print("Experiment: Basic 2D Digit Descent")
    print("=" * 60)

    true_coeffs = [3, 5]
    num_digits = 3  # 0-7 range
    p = 2
    n_points = 4

    data = generate_perfect_data(true_coeffs, n_points)
    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Data: {[(pt.x, pt.y) for pt in data]}")

    # Brute force
    print("\n--- Brute Force ---")
    bf_coeffs, bf_int, bf_val = brute_force_multidim(data, num_digits, 2, p=p)
    print(f"Result: {bf_coeffs}, val={bf_val}")

    # Digit descent
    print("\n--- Digit Descent ---")
    dd_coeffs, dd_int, stats = digit_descent_regression(
        data, 2, num_digits, p=p, n_samples=10, verbose=True
    )
    dd_val = multidim_valuation_sum(data, dd_coeffs, dd_int, p)
    print(f"\nResult: {dd_coeffs}, val={dd_val}")
    print(f"Evaluations: {stats['evaluations']}")
    print(f"Correct: {dd_coeffs == bf_coeffs}")


def experiment_accuracy_scaling():
    """Test accuracy across dimensions and digits."""
    print("\n" + "=" * 60)
    print("Experiment: Accuracy Scaling")
    print("=" * 60)

    p = 2
    n_points = 5
    n_trials = 20
    n_samples = 20

    print(f"\nPrime p={p}, points={n_points}, samples={n_samples}, trials={n_trials}")
    print(f"\n{'Dims':>4} | {'Digits':>6} | {'Accuracy':>10} | {'Evals':>10} | {'BF Evals':>12}")
    print("-" * 55)

    for n_dims in [2, 3, 4]:
        for num_digits in [3, 4, 5]:
            correct = 0
            total_evals = 0

            for trial in range(n_trials):
                # Random true coefficients
                max_coeff = (1 << num_digits) - 1
                true_coeffs = [random.randint(1, max_coeff) for _ in range(n_dims)]
                data = generate_perfect_data(true_coeffs, n_points)

                # Brute force (ground truth)
                bf_coeffs, _, _ = brute_force_multidim(data, num_digits, n_dims, p=p)

                # Digit descent
                dd_coeffs, _, stats = digit_descent_regression(
                    data, n_dims, num_digits, p=p, n_samples=n_samples
                )
                total_evals += stats['evaluations']

                if dd_coeffs == list(bf_coeffs):
                    correct += 1

            avg_evals = total_evals / n_trials
            bf_evals = p ** (num_digits * n_dims)
            accuracy = 100 * correct / n_trials

            print(f"{n_dims:>4} | {num_digits:>6} | {accuracy:>9.0f}% | {avg_evals:>10.0f} | {bf_evals:>12,}")


def experiment_with_refinement():
    """Test digit descent with refinement passes."""
    print("\n" + "=" * 60)
    print("Experiment: Digit Descent with Refinement")
    print("=" * 60)

    true_coeffs = [5, 7, 3]
    num_digits = 4
    p = 2
    n_points = 5
    n_trials = 20

    print(f"\nTrue coefficients: {true_coeffs}")
    print(f"Digits: {num_digits}, Prime: {p}")

    data = generate_perfect_data(true_coeffs, n_points)

    # Without refinement
    correct_no_refine = 0
    for _ in range(n_trials):
        data = generate_perfect_data(true_coeffs, n_points)
        bf_coeffs, _, _ = brute_force_multidim(data, num_digits, 3, p=p)
        dd_coeffs, _, _ = digit_descent_regression(data, 3, num_digits, p=p, n_samples=10)
        if dd_coeffs == list(bf_coeffs):
            correct_no_refine += 1

    # With refinement
    correct_with_refine = 0
    for _ in range(n_trials):
        data = generate_perfect_data(true_coeffs, n_points)
        bf_coeffs, _, _ = brute_force_multidim(data, num_digits, 3, p=p)
        dd_coeffs, _, _ = digit_descent_with_refinement(
            data, 3, num_digits, p=p, n_samples=10, n_refinement_passes=2
        )
        if dd_coeffs == list(bf_coeffs):
            correct_with_refine += 1

    print(f"\nWithout refinement: {correct_no_refine}/{n_trials} ({100*correct_no_refine/n_trials:.0f}%)")
    print(f"With refinement:    {correct_with_refine}/{n_trials} ({100*correct_with_refine/n_trials:.0f}%)")


def experiment_speedup():
    """Compare runtime of digit descent vs brute force."""
    print("\n" + "=" * 60)
    print("Experiment: Speedup Analysis")
    print("=" * 60)

    p = 2
    n_points = 4
    n_samples = 15

    print(f"\n{'Dims':>4} | {'Digits':>6} | {'Search Space':>14} | {'BF Time':>10} | {'DD Time':>10} | {'Speedup':>10}")
    print("-" * 75)

    for n_dims in [2, 3]:
        for num_digits in [4, 5, 6, 7]:
            max_coeff = (1 << num_digits) - 1
            true_coeffs = [random.randint(1, max_coeff) for _ in range(n_dims)]
            data = generate_perfect_data(true_coeffs, n_points)

            search_space = p ** (num_digits * n_dims)

            # Skip very large brute force
            if search_space > 1_000_000:
                bf_time = float('inf')
            else:
                start = time.time()
                brute_force_multidim(data, num_digits, n_dims, p=p)
                bf_time = time.time() - start

            # Digit descent timing
            start = time.time()
            digit_descent_regression(data, n_dims, num_digits, p=p, n_samples=n_samples)
            dd_time = time.time() - start

            if bf_time == float('inf'):
                speedup = "N/A"
                bf_str = ">1s"
            else:
                speedup = f"{bf_time/dd_time:.1f}x"
                bf_str = f"{bf_time:.4f}s"

            print(f"{n_dims:>4} | {num_digits:>6} | {search_space:>14,} | {bf_str:>10} | {dd_time:>9.4f}s | {speedup:>10}")


def experiment_different_primes():
    """Test with different primes (3-adic, 5-adic)."""
    print("\n" + "=" * 60)
    print("Experiment: Different Primes")
    print("=" * 60)

    n_points = 5
    n_trials = 20
    n_samples = 20

    for p in [2, 3, 5]:
        print(f"\n--- Prime p={p} ---")

        for n_dims in [2, 3]:
            num_digits = 3  # p^3 range for each coefficient

            correct = 0
            for trial in range(n_trials):
                max_coeff = p ** num_digits - 1
                true_coeffs = [random.randint(1, max_coeff) for _ in range(n_dims)]
                data = generate_perfect_data(true_coeffs, n_points)

                bf_coeffs, _, _ = brute_force_multidim(data, num_digits, n_dims, p=p)
                dd_coeffs, _, _ = digit_descent_regression(
                    data, n_dims, num_digits, p=p, n_samples=n_samples
                )

                if dd_coeffs == list(bf_coeffs):
                    correct += 1

            accuracy = 100 * correct / n_trials
            hypercube_size = p ** n_dims
            print(f"  {n_dims}D: {accuracy:.0f}% (hypercube size: {hypercube_size})")


def experiment_noisy_data():
    """Test with noisy data."""
    print("\n" + "=" * 60)
    print("Experiment: Noisy Data")
    print("=" * 60)

    true_coeffs = [5, 7]
    num_digits = 4
    p = 2
    n_points = 6
    n_trials = 20

    print(f"True coefficients: {true_coeffs}")

    for noise_bits in [0, 1, 2]:
        correct = 0
        for _ in range(n_trials):
            data = generate_noisy_data(true_coeffs, n_points, noise_bits=noise_bits)
            bf_coeffs, _, _ = brute_force_multidim(data, num_digits, 2, p=p)
            dd_coeffs, _, _ = digit_descent_with_refinement(
                data, 2, num_digits, p=p, n_samples=15
            )
            if dd_coeffs == list(bf_coeffs):
                correct += 1

        print(f"  Noise bits={noise_bits}: {correct}/{n_trials} ({100*correct/n_trials:.0f}%)")


def run_all_experiments():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("DIGIT DESCENT P-ADIC REGRESSION EXPERIMENTS")
    print("=" * 70)

    experiment_basic_2d()
    experiment_accuracy_scaling()
    experiment_with_refinement()
    experiment_speedup()
    experiment_different_primes()
    experiment_noisy_data()


if __name__ == "__main__":
    run_all_experiments()
