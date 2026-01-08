#!/usr/bin/env python3
"""
Classical Monte Carlo Ladder Algorithm

Instead of quantum superposition, randomly sample tournament brackets
and track the most common winner.

Approaches:
1. Random Tournament: Start with random coefficient, run through rounds
2. Random Walk: Each round, randomly pick a starting pair to evaluate
"""

import random
from collections import Counter
from typing import List, Tuple, Dict

from padic_core import DataPoint, classical_valuation_sum


def tournament_winner(data: List[DataPoint], a: int, a_prime: int) -> int:
    """
    Compare two coefficients and return the winner.
    Winner has higher valuation sum (better p-adic fit).
    Ties go to the smaller coefficient.
    """
    f_a = classical_valuation_sum(data, a, 0)
    f_a_prime = classical_valuation_sum(data, a_prime, 0)

    if f_a >= f_a_prime:
        return a
    else:
        return a_prime


def run_single_tournament(data: List[DataPoint], coeff_bits: int, start_a: int) -> int:
    """
    Run one full tournament starting from coefficient 'start_a'.

    For each round k (1 to coeff_bits):
    - Compare current coefficient with partner (differs in bit k-1)
    - Winner advances to next round

    Returns the final winner.
    """
    current = start_a

    for round_num in range(1, coeff_bits + 1):
        bit_to_flip = round_num - 1
        partner = current ^ (1 << bit_to_flip)
        current = tournament_winner(data, current, partner)

    return current


def monte_carlo_ladder(data: List[DataPoint], coeff_bits: int,
                       n_samples: int = 100) -> Tuple[int, Dict[int, int]]:
    """
    Monte Carlo ladder algorithm.

    1. Sample n_samples random starting coefficients
    2. Run each through the full tournament
    3. Return the most common winner and winner counts

    Returns: (most_common_winner, {winner: count})
    """
    max_coeff = (1 << coeff_bits) - 1
    winner_counts = Counter()

    for _ in range(n_samples):
        # Random starting coefficient
        start_a = random.randint(0, max_coeff)

        # Run tournament
        winner = run_single_tournament(data, coeff_bits, start_a)
        winner_counts[winner] += 1

    # Most common winner
    most_common = winner_counts.most_common(1)[0][0]

    return most_common, dict(winner_counts)


def brute_force_optimal(data: List[DataPoint], coeff_bits: int) -> Tuple[int, int]:
    """
    Find optimal coefficient by checking all possibilities.
    Returns (optimal_a, max_valuation_sum).
    """
    best_a = 0
    best_sum = -float('inf')

    for a in range(1 << coeff_bits):
        val_sum = classical_valuation_sum(data, a, 0)
        if val_sum > best_sum:
            best_sum = val_sum
            best_a = a

    return best_a, best_sum


def analyze_tournament_structure(data: List[DataPoint], coeff_bits: int):
    """
    Analyze which starting coefficients lead to which winners.
    This helps understand the tournament structure.
    """
    print(f"\n=== Tournament Structure Analysis ({coeff_bits}-bit) ===")

    # Show valuation sums
    print("\nValuation sums:")
    for a in range(1 << coeff_bits):
        f_a = classical_valuation_sum(data, a, 0)
        print(f"  a={a} (bin={bin(a)}): F(a)={f_a}")

    # Show which starting points lead to which winners
    print("\nTournament outcomes:")
    winner_map = {}
    for start in range(1 << coeff_bits):
        winner = run_single_tournament(data, coeff_bits, start)
        if winner not in winner_map:
            winner_map[winner] = []
        winner_map[winner].append(start)

    for winner, starters in sorted(winner_map.items()):
        print(f"  Winner {winner}: starting points {starters}")

    return winner_map


def experiment_accuracy_vs_samples(data: List[DataPoint], coeff_bits: int,
                                    sample_counts: List[int], n_trials: int = 50):
    """
    Test how accuracy improves with more samples.
    """
    print(f"\n=== Accuracy vs Samples ({coeff_bits}-bit) ===")

    optimal, _ = brute_force_optimal(data, coeff_bits)
    print(f"Optimal coefficient: {optimal}")

    for n_samples in sample_counts:
        correct = 0
        for _ in range(n_trials):
            winner, _ = monte_carlo_ladder(data, coeff_bits, n_samples)
            if winner == optimal:
                correct += 1

        accuracy = 100 * correct / n_trials
        print(f"  {n_samples:4d} samples: {accuracy:5.1f}% accuracy")


def experiment_scaling(coeff_bits_range: List[int], n_samples: int = 100, n_trials: int = 50):
    """
    Test how the algorithm scales with coefficient bits.
    """
    print(f"\n=== Scaling Experiment ===")
    print(f"Samples per trial: {n_samples}, Trials: {n_trials}")

    for coeff_bits in coeff_bits_range:
        # Generate data with known optimal
        true_a = (1 << coeff_bits) // 2 + 1  # Pick a coefficient
        data = [
            DataPoint(1, true_a),
            DataPoint(2, 2 * true_a)
        ]

        optimal, opt_val = brute_force_optimal(data, coeff_bits)

        correct = 0
        for _ in range(n_trials):
            winner, _ = monte_carlo_ladder(data, coeff_bits, n_samples)
            if winner == optimal:
                correct += 1

        accuracy = 100 * correct / n_trials
        search_space = 1 << coeff_bits
        print(f"  {coeff_bits}-bit ({search_space:3d} coeffs): optimal={optimal}, accuracy={accuracy:.1f}%")


def experiment_sample_efficiency():
    """
    Compare samples needed vs brute force evaluations.
    """
    print("\n=== Sample Efficiency ===")

    for coeff_bits in [2, 3, 4, 5]:
        # Generate test data
        true_a = (1 << coeff_bits) // 2 + 1
        data = [DataPoint(1, true_a), DataPoint(2, 2 * true_a)]

        optimal, _ = brute_force_optimal(data, coeff_bits)
        search_space = 1 << coeff_bits

        # Find minimum samples for 90% accuracy
        for n_samples in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            correct = sum(1 for _ in range(100)
                         if monte_carlo_ladder(data, coeff_bits, n_samples)[0] == optimal)
            if correct >= 90:
                ratio = n_samples / search_space
                print(f"  {coeff_bits}-bit: {search_space:3d} coeffs, need ~{n_samples:3d} samples for 90% ({ratio:.2f}x)")
                break
        else:
            print(f"  {coeff_bits}-bit: {search_space:3d} coeffs, need >256 samples for 90%")


def main():
    print("=" * 60)
    print("Classical Monte Carlo Ladder Algorithm")
    print("=" * 60)

    # Test case: y = 2x
    data = [DataPoint(1, 2), DataPoint(2, 4)]
    coeff_bits = 2

    print(f"\nTest data: {[(pt.x, pt.y) for pt in data]}")
    print(f"Coefficient bits: {coeff_bits}")

    # Analyze tournament structure
    winner_map = analyze_tournament_structure(data, coeff_bits)

    # Run Monte Carlo
    print("\n=== Monte Carlo Results ===")
    for n_samples in [10, 50, 100, 500]:
        winner, counts = monte_carlo_ladder(data, coeff_bits, n_samples)
        optimal, _ = brute_force_optimal(data, coeff_bits)
        print(f"\n{n_samples} samples:")
        print(f"  Winner counts: {counts}")
        print(f"  Most common: {winner}")
        print(f"  Correct: {winner == optimal}")

    # Accuracy vs samples
    experiment_accuracy_vs_samples(data, coeff_bits, [5, 10, 20, 50, 100])

    # Scaling
    experiment_scaling([2, 3, 4, 5], n_samples=50, n_trials=50)

    # Sample efficiency
    experiment_sample_efficiency()

    # Test with 3-bit case
    print("\n" + "=" * 60)
    print("3-bit Test Case")
    print("=" * 60)

    data3 = [DataPoint(1, 5), DataPoint(2, 10)]
    analyze_tournament_structure(data3, 3)
    experiment_accuracy_vs_samples(data3, 3, [5, 10, 20, 50, 100, 200])


if __name__ == "__main__":
    main()


def experiment_harder_cases():
    """
    Test with cases where the optimal might be harder to find.
    """
    print("\n" + "=" * 60)
    print("Harder Test Cases")
    print("=" * 60)

    # Case 1: Noisy data
    print("\n--- Case 1: Noisy data ---")
    data = [DataPoint(1, 5), DataPoint(2, 11)]  # y ≈ 5x + noise
    analyze_tournament_structure(data, 3)

    # Case 2: Multiple good fits
    print("\n--- Case 2: Multiple points ---")
    data = [DataPoint(1, 3), DataPoint(2, 6), DataPoint(3, 9)]
    analyze_tournament_structure(data, 3)

    # Case 3: Sparse data with ties possible
    print("\n--- Case 3: Single point ---")
    data = [DataPoint(2, 6)]  # Only one point, multiple coefficients might be equally good
    analyze_tournament_structure(data, 3)

    # Case 4: Random data
    print("\n--- Case 4: Random data ---")
    import random
    random.seed(42)
    data = [DataPoint(random.randint(1, 8), random.randint(1, 100)) for _ in range(3)]
    print(f"Data: {[(p.x, p.y) for p in data]}")
    analyze_tournament_structure(data, 4)


def experiment_multiple_attractors():
    """
    Look for cases with multiple tournament attractors.
    """
    print("\n" + "=" * 60)
    print("Searching for Multiple Attractors")
    print("=" * 60)

    import random
    random.seed(123)

    for trial in range(20):
        # Random data
        n_points = random.randint(1, 4)
        data = [DataPoint(random.randint(1, 8), random.randint(1, 50))
                for _ in range(n_points)]

        coeff_bits = 3

        # Find all winners from all starting points
        winners = set()
        for start in range(1 << coeff_bits):
            w = run_single_tournament(data, coeff_bits, start)
            winners.add(w)

        if len(winners) > 1:
            print(f"\nFound {len(winners)} attractors!")
            print(f"Data: {[(p.x, p.y) for p in data]}")
            analyze_tournament_structure(data, coeff_bits)
            break
    else:
        print("No multiple attractors found in 20 random trials")

    # Systematic search for ties
    print("\n--- Searching for tie conditions ---")
    for true_a in range(8):
        data = [DataPoint(1, true_a)]
        winners = set()
        for start in range(8):
            w = run_single_tournament(data, 3, start)
            winners.add(w)
        if len(winners) > 1:
            print(f"true_a={true_a}: winners={winners}")


if __name__ == "__main__":
    experiment_harder_cases()
    experiment_multiple_attractors()
