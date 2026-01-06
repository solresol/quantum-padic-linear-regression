#!/usr/bin/env python3
"""
Comprehensive experiments for the quantum p-adic ladder algorithm.

Tests:
1. Correctness: Does the algorithm find the optimal coefficient?
2. Success probability: How often is the optimal found?
3. Comparison with classical brute force
4. Scaling behavior with problem size
"""

from qiskit import transpile
from qiskit_aer import Aer
import math
import random
import time
from typing import List, Tuple, Dict

from padic_core import DataPoint, classical_valuation_sum
from quantum_ladder_full import FullQuantumLadder, compute_quantum_valuation_sum


def generate_linear_data(true_a: int, n_points: int, x_range: int = 4) -> List[DataPoint]:
    """Generate y = a*x data with random x values."""
    data = []
    for _ in range(n_points):
        x = random.randint(1, x_range)
        y = true_a * x
        data.append(DataPoint(x, y))
    return data


def generate_noisy_data(true_a: int, n_points: int, x_range: int = 4, noise_bits: int = 1) -> List[DataPoint]:
    """Generate y = a*x + noise with small random perturbations."""
    data = []
    for _ in range(n_points):
        x = random.randint(1, x_range)
        noise = random.randint(-(1 << noise_bits), (1 << noise_bits))
        y = true_a * x + noise
        data.append(DataPoint(x, y))
    return data


def classical_brute_force(data: List[DataPoint], coeff_bits: int) -> Tuple[int, int]:
    """
    Find optimal coefficient by brute force search.
    Returns (optimal_a, max_valuation_sum).

    Note: We search for slope a only, with intercept b=0 (line through origin).
    """
    best_a = 0
    best_sum = -float('inf')

    for a in range(1 << coeff_bits):
        val_sum = classical_valuation_sum(data, a, 0)  # b=0 for lines through origin
        if val_sum > best_sum:
            best_sum = val_sum
            best_a = a

    return best_a, best_sum


def test_single_case(data: List[DataPoint], coeff_bits: int, true_a: int = None,
                     shots: int = 1024, verbose: bool = True) -> Dict:
    """
    Run quantum algorithm on a single test case.

    Returns dict with results.
    """
    result = {
        'data': data,
        'coeff_bits': coeff_bits,
        'true_a': true_a,
        'shots': shots,
    }

    # Classical brute force to find optimal
    classical_optimal, classical_val_sum = classical_brute_force(data, coeff_bits)
    result['classical_optimal'] = classical_optimal
    result['classical_val_sum'] = classical_val_sum

    if verbose:
        print(f"\nData: {[(pt.x, pt.y) for pt in data]}")
        print(f"Classical optimal: a*={classical_optimal} with valuation_sum={classical_val_sum}")

    # Run quantum algorithm
    try:
        ladder = FullQuantumLadder(data, coeff_bits=coeff_bits)

        start_time = time.time()
        quantum_winner, counts = ladder.run(round_num=coeff_bits, shots=shots)
        elapsed = time.time() - start_time

        result['quantum_winner'] = quantum_winner
        result['counts'] = counts
        result['elapsed_time'] = elapsed
        result['correct'] = (quantum_winner == classical_optimal)

        # Calculate success probability
        optimal_count = 0
        for bitstring, count in counts.items():
            val = int(bitstring.replace(' ', ''), 2)
            if val == classical_optimal:
                optimal_count += count
        result['success_probability'] = optimal_count / shots

        if verbose:
            print(f"Quantum winner: {quantum_winner}")
            print(f"Correct: {result['correct']}")
            print(f"Success probability: {result['success_probability']:.1%}")
            print(f"Time: {elapsed:.2f}s")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Error: {e}")

    return result


def experiment_correctness(n_trials: int = 10, coeff_bits: int = 2, n_points: int = 2):
    """
    Test correctness across multiple random instances.
    """
    print("=" * 60)
    print(f"Correctness Experiment: {n_trials} trials")
    print(f"Settings: {coeff_bits}-bit coefficients, {n_points} data points")
    print("=" * 60)

    correct_count = 0
    total_success_prob = 0

    for trial in range(n_trials):
        # Random true coefficient
        true_a = random.randint(0, (1 << coeff_bits) - 1)
        data = generate_linear_data(true_a, n_points, x_range=4)

        print(f"\nTrial {trial + 1}/{n_trials}: true_a={true_a}")
        result = test_single_case(data, coeff_bits, true_a, verbose=False)

        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue

        if result['correct']:
            correct_count += 1
            print(f"  CORRECT: quantum={result['quantum_winner']}, classical={result['classical_optimal']}, P={result['success_probability']:.1%}")
        else:
            print(f"  WRONG: quantum={result['quantum_winner']}, classical={result['classical_optimal']}, P={result['success_probability']:.1%}")

        total_success_prob += result['success_probability']

    print(f"\n--- Summary ---")
    print(f"Correctness: {correct_count}/{n_trials} ({100*correct_count/n_trials:.0f}%)")
    print(f"Average success probability: {100*total_success_prob/n_trials:.1f}%")


def experiment_scaling(coeff_bits_range: List[int] = [2, 3], n_points: int = 2):
    """
    Test how performance scales with coefficient bits.
    """
    print("=" * 60)
    print("Scaling Experiment")
    print("=" * 60)

    for coeff_bits in coeff_bits_range:
        print(f"\n{coeff_bits}-bit coefficients:")

        # Use perfect data for scaling test
        true_a = (1 << coeff_bits) // 2  # Middle value
        data = generate_linear_data(true_a, n_points, x_range=4)

        result = test_single_case(data, coeff_bits, true_a, shots=1024, verbose=True)


def experiment_probability_analysis(coeff_bits: int = 2, n_points: int = 2, shots: int = 4096):
    """
    Detailed analysis of measurement probability distribution.
    """
    print("=" * 60)
    print("Probability Distribution Analysis")
    print(f"Settings: {coeff_bits}-bit coefficients, {n_points} data points, {shots} shots")
    print("=" * 60)

    # Use a known good case
    true_a = 2
    data = [DataPoint(1, 2), DataPoint(2, 4)]

    print(f"\nData: {[(pt.x, pt.y) for pt in data]}")
    print(f"True coefficient: a*={true_a}")

    # Show all valuation sums
    ladder = FullQuantumLadder(data, coeff_bits=coeff_bits)
    print(f"\nValuation sums for each coefficient:")
    for a in range(1 << coeff_bits):
        val_sum = compute_quantum_valuation_sum(data, a, ladder.residual_bits)
        residuals = [pt.y - a * pt.x for pt in data]
        print(f"  a={a}: residuals={residuals}, F(a)={val_sum}")

    # Run quantum algorithm
    quantum_winner, counts = ladder.run(round_num=coeff_bits, shots=shots)

    # Analyze distribution
    print(f"\nMeasurement distribution ({shots} shots):")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for bitstring, count in sorted_counts:
        val = int(bitstring.replace(' ', ''), 2)
        prob = count / shots
        bar = '*' * int(prob * 50)
        print(f"  a={val}: {count:4d} ({prob:5.1%}) {bar}")

    # Theoretical analysis
    print("\nTheoretical analysis:")
    print("For round k, pairs (a, a') differ in bit k-1")
    print("Winner gets both outcomes' probability mass")

    bit_to_flip = coeff_bits - 1
    for a in range(1 << coeff_bits):
        if a & (1 << bit_to_flip):
            continue  # Only show each pair once
        a_prime = a ^ (1 << bit_to_flip)
        f_a = compute_quantum_valuation_sum(data, a, ladder.residual_bits)
        f_a_prime = compute_quantum_valuation_sum(data, a_prime, ladder.residual_bits)
        winner = a if f_a >= f_a_prime else a_prime
        print(f"  Pair ({a}, {a_prime}): F({a})={f_a}, F({a_prime})={f_a_prime} -> winner={winner}")


def experiment_multi_round():
    """
    Test the multi-round ladder approach.

    For k-bit coefficients:
    - Round 1: Compare bit 0 (pairs differ in LSB)
    - Round 2: Compare bit 1
    - ...
    - Round k: Compare bit k-1 (MSB)

    Each round halves the search space.
    """
    print("=" * 60)
    print("Multi-Round Ladder Experiment")
    print("=" * 60)

    coeff_bits = 3
    true_a = 5  # Binary: 101
    data = [DataPoint(1, 5), DataPoint(2, 10)]

    print(f"\nData: {[(pt.x, pt.y) for pt in data]}")
    print(f"True coefficient: a*={true_a} (binary: {bin(true_a)})")

    ladder = FullQuantumLadder(data, coeff_bits=coeff_bits)

    # Show all valuation sums
    print(f"\nValuation sums:")
    for a in range(1 << coeff_bits):
        val_sum = compute_quantum_valuation_sum(data, a, ladder.residual_bits)
        print(f"  a={a} ({bin(a)}): F(a)={val_sum}")

    # Run each round
    for round_num in range(1, coeff_bits + 1):
        print(f"\n--- Round {round_num} (flip bit {round_num - 1}) ---")

        quantum_winner, counts = ladder.run(round_num=round_num, shots=1024)

        # Show expected winners for this round
        bit_to_flip = round_num - 1
        print("Expected winners:")
        for a in range(1 << coeff_bits):
            if a & (1 << bit_to_flip):
                continue
            a_prime = a ^ (1 << bit_to_flip)
            f_a = compute_quantum_valuation_sum(data, a, ladder.residual_bits)
            f_a_prime = compute_quantum_valuation_sum(data, a_prime, ladder.residual_bits)
            winner = a if f_a >= f_a_prime else a_prime
            print(f"  ({a}, {a_prime}): F={f_a}, F'={f_a_prime} -> {winner}")

        print(f"\nQuantum results:")
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        for bitstring, count in sorted_counts[:4]:
            val = int(bitstring.replace(' ', ''), 2)
            print(f"  s={val}: {count} shots ({100*count/1024:.1f}%)")


def run_all_experiments():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("QUANTUM P-ADIC LADDER ALGORITHM EXPERIMENTS")
    print("=" * 70)

    # Basic correctness
    experiment_correctness(n_trials=5, coeff_bits=2, n_points=2)

    # Probability analysis
    experiment_probability_analysis(coeff_bits=2, n_points=2, shots=1024)

    # Scaling
    experiment_scaling(coeff_bits_range=[2, 3], n_points=2)

    # Multi-round
    experiment_multi_round()


if __name__ == "__main__":
    # Quick test first
    print("Quick sanity check...")
    data = [DataPoint(1, 2), DataPoint(2, 4)]
    test_single_case(data, coeff_bits=2, true_a=2, shots=512, verbose=True)

    print("\n" + "=" * 70)
    print("Running full experiment suite...")
    print("=" * 70)

    run_all_experiments()
