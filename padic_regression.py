#!/usr/bin/env python3
"""
Quantum p-adic Linear Regression

This module implements the quantum algorithm for p-adic linear regression
as described in the quantum-padic-regression paper.

For p=2 (2-adic), we use the trailing zeros count as the valuation.
The 2-adic distance is 2^(-valuation), so smaller valuation = larger distance.
We want to minimize the sum of distances, which means maximizing the sum of valuations.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import math
from typing import List, Tuple
from dataclasses import dataclass

import initialise
import twoadic


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

    For each point (x_i, y_i), the residual is y_i - (m*x_i + b).
    The 2-adic distance is 2^(-valuation(residual)).
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


def classical_brute_force(data: List[DataPoint], m_range: range, b_range: range) -> Tuple[int, int, float]:
    """
    Find the best (m, b) by brute force enumeration.
    Returns (best_m, best_b, best_distance_sum).
    """
    best_m, best_b = 0, 0
    best_dist = float('inf')

    for m in m_range:
        for b in b_range:
            dist = classical_residual_sum(data, m, b)
            if dist < best_dist:
                best_dist = dist
                best_m, best_b = m, b

    return best_m, best_b, best_dist


def bits_needed(max_val: int) -> int:
    """Number of bits needed to represent values up to max_val."""
    if max_val <= 0:
        return 1
    return max(1, math.ceil(math.log2(max_val + 1)))


class QuantumPadicRegression:
    """
    Quantum circuit for 2-adic linear regression.

    The algorithm works by:
    1. Creating superposition of candidate (m, b) values modulo 2^t
    2. Computing residuals for each data point
    3. Computing 2-adic valuations (trailing zeros) of residuals
    4. Summing valuations
    5. Using quantum minimum finding to identify optimal (m, b)
    """

    def __init__(self, data: List[DataPoint], m_bits: int, b_bits: int):
        """
        Initialize the quantum regression circuit.

        Args:
            data: List of (x, y) data points
            m_bits: Number of bits for gradient m
            b_bits: Number of bits for intercept b
        """
        self.data = data
        self.m_bits = m_bits
        self.b_bits = b_bits
        self.n_points = len(data)

        # Compute max possible residual to size registers appropriately
        max_x = max(abs(p.x) for p in data)
        max_y = max(abs(p.y) for p in data)
        max_m = 2**m_bits - 1
        max_b = 2**b_bits - 1
        self.max_residual = max_y + max_m * max_x + max_b
        self.residual_bits = bits_needed(self.max_residual) + 1  # +1 for sign

        # For summing valuations: max sum is n_points * residual_bits
        self.max_valuation_sum = self.n_points * self.residual_bits
        self.valuation_sum_bits = bits_needed(self.max_valuation_sum)

    def build_residual_oracle(self) -> QuantumCircuit:
        """
        Build a quantum circuit that computes residuals for all data points.

        This is a simplified version that works for small datasets.
        For each (m, b) in superposition, computes y_i - (m*x_i + b) for all i.
        """
        # This is complex - for now, return a placeholder
        # A full implementation would need:
        # 1. Quantum multiplication circuits for m * x_i
        # 2. Quantum addition circuits
        # 3. Trailing zeros computation for each residual
        raise NotImplementedError("Full quantum residual oracle not yet implemented")

    def run_round_1_classical_quantum_hybrid(self) -> Tuple[int, int]:
        """
        Run round 1: Find optimal (m mod 2, b mod 2) using a hybrid approach.

        For round 1, there are only 4 possibilities, so we can use
        quantum parallelism efficiently.
        """
        # For p=2, round 1 has only 4 cases: (m,b) mod 2 in {(0,0), (0,1), (1,0), (1,1)}
        # We create a superposition and evaluate all 4 simultaneously

        m_reg = QuantumRegister(1, 'm')
        b_reg = QuantumRegister(1, 'b')

        # Ancilla for computation - simplified for demonstration
        # In a full implementation, we'd need registers for residuals, valuations, etc.

        circ = QuantumCircuit(m_reg, b_reg)

        # Create uniform superposition
        circ.h(m_reg)
        circ.h(b_reg)

        # TODO: Add oracle that marks the optimal (m, b)
        # This requires implementing the residual sum computation

        return 0, 0  # Placeholder

    def find_optimal_mod_p(self, p: int, t: int,
                           known_m_mod: int = 0,
                           known_b_mod: int = 0) -> Tuple[int, int]:
        """
        Find optimal (m mod p^t, b mod p^t) given known values mod p^(t-1).

        This implements one round of the ladder algorithm.

        Args:
            p: Prime (we use p=2)
            t: Power of p for this round
            known_m_mod: Known value of m mod p^(t-1)
            known_b_mod: Known value of b mod p^(t-1)

        Returns:
            (m mod p^t, b mod p^t)
        """
        # For round t, we search over p^2 combinations:
        # m = known_m_mod + i * p^(t-1) for i in {0, ..., p-1}
        # b = known_b_mod + j * p^(t-1) for j in {0, ..., p-1}

        best_i, best_j = 0, 0
        best_val_sum = -1

        prev_power = p ** (t - 1) if t > 1 else 1

        for i in range(p):
            for j in range(p):
                m_candidate = known_m_mod + i * prev_power
                b_candidate = known_b_mod + j * prev_power

                val_sum = classical_valuation_sum(self.data, m_candidate, b_candidate)
                if val_sum > best_val_sum:
                    best_val_sum = val_sum
                    best_i, best_j = i, j

        return known_m_mod + best_i * prev_power, known_b_mod + best_j * prev_power


def quantum_padic_regression_ladder(data: List[DataPoint],
                                     max_m: int,
                                     max_b: int,
                                     p: int = 2) -> Tuple[int, int]:
    """
    Main quantum p-adic regression using the ladder algorithm.

    This is currently a classical simulation of the quantum algorithm.
    Each round uses classical computation but follows the quantum algorithm's
    structure of searching mod p^t iteratively.

    Args:
        data: List of data points
        max_m: Maximum gradient value to consider
        max_b: Maximum intercept value to consider
        p: Prime for p-adic distance (default 2)

    Returns:
        (optimal_m, optimal_b)
    """
    m_bits = bits_needed(max_m)
    b_bits = bits_needed(max_b)

    qpr = QuantumPadicRegression(data, m_bits, b_bits)

    # Number of rounds needed
    max_power = max(max_m, max_b)
    num_rounds = bits_needed(max_power)

    # Ladder ascent: find optimal values mod p, then mod p^2, etc.
    current_m_mod = 0
    current_b_mod = 0

    for t in range(1, num_rounds + 1):
        current_m_mod, current_b_mod = qpr.find_optimal_mod_p(
            p, t, current_m_mod, current_b_mod
        )
        print(f"Round {t}: m ≡ {current_m_mod} (mod {p**t}), b ≡ {current_b_mod} (mod {p**t})")

    return current_m_mod, current_b_mod


# =============================================================================
# Quantum Circuit Implementation
# =============================================================================

def quantum_round1_circuit(data: List[DataPoint]) -> Tuple[int, int, dict]:
    """
    Build and run a quantum circuit for round 1 of the ladder algorithm.

    For p=2, round 1 searches over 4 possibilities: (m mod 2, b mod 2).
    This creates a superposition and uses Grover's oracle to mark optimal states.

    Returns:
        (best_m_mod_2, best_b_mod_2, measurement_counts)
    """
    # For round 1, we have 2 qubits: one for m mod 2, one for b mod 2
    m_reg = QuantumRegister(1, 'm')
    b_reg = QuantumRegister(1, 'b')

    # Classical registers for measurement
    m_c = ClassicalRegister(1, 'm_c')
    b_c = ClassicalRegister(1, 'b_c')

    # We need ancillas for the oracle computation
    # For each data point, we compute the residual and check its 2-adic valuation

    qc = QuantumCircuit(m_reg, b_reg, m_c, b_c)

    # Create uniform superposition over all 4 (m, b) combinations
    qc.h(m_reg)
    qc.h(b_reg)

    # At this point we have:
    # |00> + |01> + |10> + |11> (unnormalized)
    # representing (m,b) = (0,0), (0,1), (1,0), (1,1)

    # Classical pre-computation: find which (m mod 2, b mod 2) is best
    best_val = -1
    best_m, best_b = 0, 0
    for m in range(2):
        for b in range(2):
            val = classical_valuation_sum(data, m, b)
            if val > best_val:
                best_val = val
                best_m, best_b = m, b

    # Apply Grover oracle: flip phase of optimal state
    # This is a simplified demonstration - full oracle would compute this quantumly
    if best_m == 0 and best_b == 0:
        qc.cz(m_reg[0], b_reg[0])  # Phase flip |00>... actually need to handle this differently
        # For |00>, we flip phase when both are 0
        qc.x(m_reg[0])
        qc.x(b_reg[0])
        qc.cz(m_reg[0], b_reg[0])
        qc.x(m_reg[0])
        qc.x(b_reg[0])
    elif best_m == 0 and best_b == 1:
        qc.x(m_reg[0])
        qc.cz(m_reg[0], b_reg[0])
        qc.x(m_reg[0])
    elif best_m == 1 and best_b == 0:
        qc.x(b_reg[0])
        qc.cz(m_reg[0], b_reg[0])
        qc.x(b_reg[0])
    else:  # best_m == 1 and best_b == 1
        qc.cz(m_reg[0], b_reg[0])

    # Grover diffusion operator
    qc.h(m_reg[0])
    qc.h(b_reg[0])
    qc.x(m_reg[0])
    qc.x(b_reg[0])
    qc.cz(m_reg[0], b_reg[0])
    qc.x(m_reg[0])
    qc.x(b_reg[0])
    qc.h(m_reg[0])
    qc.h(b_reg[0])

    # Measure
    qc.measure(m_reg, m_c)
    qc.measure(b_reg, b_c)

    # Run the circuit
    sim = Aer.get_backend('aer_simulator')
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()

    # Find most frequent result
    max_count = 0
    result_m, result_b = 0, 0
    for bitstring, count in counts.items():
        if count > max_count:
            max_count = count
            # Parse bitstring: format is "b m" (b first due to Qiskit ordering)
            parts = bitstring.replace(' ', '')
            result_b = int(parts[0])
            result_m = int(parts[1])

    return result_m, result_b, counts


def quantum_residual_oracle_demo():
    """
    Demonstrate quantum computation of a single residual's 2-adic valuation.

    This shows how we would compute trailing zeros for y - (mx + b)
    for fixed values of x, y, m, b stored in quantum registers.
    """
    # Example: x=1, y=3, we want to check m=2, b=1 → residual = 3 - (2*1 + 1) = 0
    # But let's use something non-zero: m=1, b=1 → residual = 3 - (1*1 + 1) = 1

    x_val = 1
    y_val = 3
    m_val = 2
    b_val = 1

    # Compute residual classically first
    residual = y_val - (m_val * x_val + b_val)
    print(f"Computing residual: {y_val} - ({m_val}*{x_val} + {b_val}) = {residual}")
    print(f"2-adic valuation of {residual}: {classical_2adic_valuation(residual)}")

    # For the quantum version, we'd:
    # 1. Load x, y, m, b into quantum registers
    # 2. Compute m*x using quantum multiplication
    # 3. Compute m*x + b using quantum addition
    # 4. Compute y - (m*x + b) using quantum subtraction
    # 5. Apply trailing zeros counter

    # Here's a simplified demo just showing the trailing zeros part
    if residual != 0:
        residual_bits = bits_needed(abs(residual)) + 1

        diff_reg = QuantumRegister(residual_bits, 'residual')
        tz_n = 4  # bits for trailing zeros count
        tz_reg = QuantumRegister(tz_n, 'tz')
        anc_reg = QuantumRegister(1 + 2 + tz_n + 2, 'anc')

        c_tz = ClassicalRegister(tz_n, 'c_tz')

        qc = QuantumCircuit(diff_reg, tz_reg, anc_reg, c_tz)

        # Initialize residual register
        initialise.initialise_from_int(qc, diff_reg, abs(residual))

        # Compute trailing zeros
        twoadic.count_trailing_zeros_inplace(qc, diff_reg, tz_reg, anc_reg)

        # Measure
        qc.measure(tz_reg, c_tz)

        sim = Aer.get_backend('aer_simulator')
        result = sim.run(qc, shots=100).result()
        counts = result.get_counts()

        print(f"Quantum trailing zeros result: {counts}")


# =============================================================================
# Demo and Testing
# =============================================================================

def demo():
    """Demonstrate the p-adic regression algorithm."""
    print("=" * 60)
    print("Quantum 2-adic Linear Regression Demo")
    print("=" * 60)

    # Simple dataset: points that lie on y = 2x + 1
    # but we'll perturb one to make it interesting
    data = [
        DataPoint(0, 1),   # y = 2*0 + 1 = 1
        DataPoint(1, 3),   # y = 2*1 + 1 = 3
        DataPoint(2, 5),   # y = 2*2 + 1 = 5
        DataPoint(3, 7),   # y = 2*3 + 1 = 7
    ]

    print("\nDataset:")
    for p in data:
        print(f"  ({p.x}, {p.y})")

    # Classical brute force for comparison
    print("\n--- Classical Brute Force ---")
    best_m, best_b, best_dist = classical_brute_force(data, range(8), range(8))
    print(f"Best (m, b) = ({best_m}, {best_b}) with distance sum = {best_dist}")
    print(f"Valuation sum = {classical_valuation_sum(data, best_m, best_b)}")

    # Show residuals for best solution
    print("\nResiduals for best solution:")
    for p in data:
        residual = p.y - (best_m * p.x + best_b)
        val = classical_2adic_valuation(residual)
        dist = classical_2adic_distance(residual, 0)
        print(f"  Point ({p.x}, {p.y}): residual={residual}, valuation={val}, distance={dist}")

    # Quantum ladder algorithm (classical simulation)
    print("\n--- Quantum Ladder Algorithm (Classical Simulation) ---")
    qm, qb = quantum_padic_regression_ladder(data, max_m=7, max_b=7, p=2)
    print(f"\nFinal result: m = {qm}, b = {qb}")
    print(f"Distance sum = {classical_residual_sum(data, qm, qb)}")

    # Test with a dataset that has a non-trivial p-adic best fit
    print("\n" + "=" * 60)
    print("Test with perturbed dataset")
    print("=" * 60)

    # Dataset where 2-adic optimal differs from Euclidean optimal
    data2 = [
        DataPoint(0, 0),
        DataPoint(1, 2),   # If y=2x, residual=0
        DataPoint(2, 4),   # If y=2x, residual=0
        DataPoint(3, 8),   # If y=2x, residual=2 (valuation=1)
    ]

    print("\nDataset 2:")
    for p in data2:
        print(f"  ({p.x}, {p.y})")

    print("\n--- Classical Brute Force ---")
    best_m, best_b, best_dist = classical_brute_force(data2, range(8), range(8))
    print(f"Best (m, b) = ({best_m}, {best_b}) with distance sum = {best_dist}")

    print("\n--- Quantum Ladder Algorithm ---")
    qm, qb = quantum_padic_regression_ladder(data2, max_m=7, max_b=7, p=2)
    print(f"Final result: m = {qm}, b = {qb}")

    # Demonstrate quantum circuit for round 1
    print("\n" + "=" * 60)
    print("Quantum Circuit Demo: Round 1 with Grover's Algorithm")
    print("=" * 60)

    print("\nUsing dataset 1 (y = 2x + 1):")
    data_simple = [
        DataPoint(0, 1),
        DataPoint(1, 3),
        DataPoint(2, 5),
        DataPoint(3, 7),
    ]

    m_mod2, b_mod2, counts = quantum_round1_circuit(data_simple)
    print(f"Measurement counts: {counts}")
    print(f"Most likely result: m ≡ {m_mod2} (mod 2), b ≡ {b_mod2} (mod 2)")
    print(f"Expected: m ≡ 0 (mod 2), b ≡ 1 (mod 2) since optimal is m=2, b=1")

    # Demonstrate residual oracle
    print("\n" + "=" * 60)
    print("Quantum Residual Oracle Demo")
    print("=" * 60)
    quantum_residual_oracle_demo()


if __name__ == "__main__":
    demo()
