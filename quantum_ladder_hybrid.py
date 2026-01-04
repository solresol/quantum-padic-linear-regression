#!/usr/bin/env python3
"""
Hybrid Quantum Ladder Algorithm

Uses classical precomputation for residual sums, but quantum circuits for:
- Superposition over coefficient values
- Comparison (quantum)
- Winner selection (quantum)
- Measurement

This verifies the algorithm structure works correctly, isolating it from
bugs in the quantum arithmetic (trailing zeros counter, etc.)

The full quantum version would replace the classical oracle with quantum
residual computation once the arithmetic primitives are fixed.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import math
from typing import List, Tuple

from padic_core import DataPoint, classical_2adic_valuation


def compute_quantum_valuation_sum(data: List[DataPoint], a: int, residual_bits: int) -> int:
    """Compute valuation sum matching what quantum circuit would produce."""
    total = 0
    for pt in data:
        residual = pt.y - a * pt.x
        if residual == 0:
            total += residual_bits
        else:
            if residual < 0:
                residual = (1 << residual_bits) + residual
            count = 0
            while residual > 0 and (residual & 1) == 0:
                count += 1
                residual >>= 1
            total += count
    return total


class HybridQuantumLadder:
    """
    Hybrid implementation: classical oracle + quantum comparison/selection.
    """

    def __init__(self, data: List[DataPoint], coeff_bits: int = 2):
        self.data = data
        self.k = coeff_bits
        self.residual_bits = 6  # Fixed for simplicity

        # Precompute valuation sums for all coefficients
        self.valuation_sums = {}
        for a in range(1 << self.k):
            self.valuation_sums[a] = compute_quantum_valuation_sum(data, a, self.residual_bits)

        print(f"Precomputed valuations: {self.valuation_sums}")

    def build_circuit(self, round_num: int) -> Tuple[QuantumCircuit, dict]:
        """
        Build circuit for round t.

        Uses classical oracle to set sum registers, then quantum comparison and copy.
        """
        bit_to_flip = round_num - 1

        # Registers
        a_reg = QuantumRegister(self.k, 'a')
        s_reg = QuantumRegister(self.k, 's')
        sum_a_reg = QuantumRegister(8, 'sum_a')  # 8 bits for valuation sum
        sum_ap_reg = QuantumRegister(8, 'sum_ap')
        cmp_reg = QuantumRegister(1, 'cmp')
        scratch = QuantumRegister(10, 'scratch')
        s_classical = ClassicalRegister(self.k, 's_c')

        circ = QuantumCircuit(a_reg, s_reg, sum_a_reg, sum_ap_reg, cmp_reg, scratch, s_classical)

        # Step 1: Superposition
        for q in a_reg:
            circ.h(q)

        circ.barrier(label='superposition')

        # Step 2: Classical oracle - set sum registers based on a value
        # For each possible a, conditionally set the sum registers
        for a in range(1 << self.k):
            a_prime = a ^ (1 << bit_to_flip)
            sum_a = self.valuation_sums[a]
            sum_ap = self.valuation_sums[a_prime]

            # Build controls for "a_reg == a"
            controls_to_flip = [i for i in range(self.k) if not ((a >> i) & 1)]

            for i in controls_to_flip:
                circ.x(a_reg[i])

            # Set sum_a_reg bits
            for bit in range(8):
                if (sum_a >> bit) & 1:
                    if self.k == 2:
                        circ.ccx(a_reg[0], a_reg[1], sum_a_reg[bit])
                    else:
                        circ.mcx([a_reg[i] for i in range(self.k)], sum_a_reg[bit])

            # Set sum_ap_reg bits
            for bit in range(8):
                if (sum_ap >> bit) & 1:
                    if self.k == 2:
                        circ.ccx(a_reg[0], a_reg[1], sum_ap_reg[bit])
                    else:
                        circ.mcx([a_reg[i] for i in range(self.k)], sum_ap_reg[bit])

            for i in controls_to_flip:
                circ.x(a_reg[i])

        circ.barrier(label='oracle')

        # Step 3: Quantum comparison - is sum_a >= sum_ap?
        # Compute sum_a - sum_ap, check sign
        self._compare_sums(circ, sum_a_reg, sum_ap_reg, cmp_reg, scratch)

        circ.barrier(label='compare')

        # Step 4: Copy winner to s
        # If cmp=1: s = a
        # If cmp=0: s = a XOR 2^bit_to_flip
        for i in range(self.k):
            circ.cx(a_reg[i], s_reg[i])

        circ.x(cmp_reg[0])
        circ.cx(cmp_reg[0], s_reg[bit_to_flip])
        circ.x(cmp_reg[0])

        circ.barrier(label='copy')

        # Step 5: Measure s
        circ.measure(s_reg, s_classical)

        return circ, {'a': a_reg, 's': s_reg, 's_classical': s_classical}

    def _compare_sums(self, circ, sum_a_reg, sum_ap_reg, cmp_reg, scratch):
        """Compare sum_a >= sum_ap using subtraction."""
        n = len(sum_a_reg)

        # Compute sum_a - sum_ap using correct in-place subtraction
        # First flip sum_ap to get ~sum_ap
        for i in range(n):
            circ.x(sum_ap_reg[i])

        # Add ~sum_ap to sum_a
        self._add_inplace(circ, sum_ap_reg, sum_a_reg, scratch)

        # Restore sum_ap
        for i in range(n):
            circ.x(sum_ap_reg[i])

        # Add 1 to complete two's complement
        self._add_one(circ, sum_a_reg)

        # Now sum_a contains sum_a - sum_ap
        # If MSB is 0, result is non-negative
        circ.x(sum_a_reg[n-1])
        circ.cx(sum_a_reg[n-1], cmp_reg[0])
        circ.x(sum_a_reg[n-1])

        # Restore sum_a by reversing the subtraction
        self._subtract_one(circ, sum_a_reg)
        for i in range(n):
            circ.x(sum_ap_reg[i])
        self._add_inplace(circ, sum_ap_reg, sum_a_reg, scratch)
        for i in range(n):
            circ.x(sum_ap_reg[i])

    def _add_inplace(self, circ, a_reg, b_reg, scratch):
        """Correct in-place addition."""
        n = min(len(a_reg), len(b_reg))

        # Compute carries first
        circ.ccx(a_reg[0], b_reg[0], scratch[0])
        for i in range(1, n - 1):
            circ.ccx(a_reg[i], b_reg[i], scratch[i])
            circ.ccx(a_reg[i], scratch[i-1], scratch[i])
            circ.ccx(b_reg[i], scratch[i-1], scratch[i])

        # Compute sums from MSB down
        for i in range(n - 1, 0, -1):
            circ.cx(a_reg[i], b_reg[i])
            circ.cx(scratch[i-1], b_reg[i])

        circ.cx(a_reg[0], b_reg[0])

        # Uncompute carries
        for i in range(n - 2, 0, -1):
            circ.ccx(b_reg[i], scratch[i-1], scratch[i])
            circ.ccx(a_reg[i], scratch[i-1], scratch[i])
            circ.ccx(a_reg[i], b_reg[i], scratch[i])

        circ.ccx(a_reg[0], b_reg[0], scratch[0])

    def _add_one(self, circ, reg):
        """Add 1 to register."""
        n = len(reg)
        for i in range(n - 1, -1, -1):
            if i == 0:
                circ.x(reg[0])
            else:
                controls = [reg[j] for j in range(i)]
                if len(controls) == 1:
                    circ.cx(controls[0], reg[i])
                elif len(controls) == 2:
                    circ.ccx(controls[0], controls[1], reg[i])
                else:
                    circ.mcx(controls, reg[i])

    def _subtract_one(self, circ, reg):
        """Subtract 1 from register (reverse of add_one)."""
        n = len(reg)
        for i in range(n):
            if i == 0:
                circ.x(reg[0])
            else:
                controls = [reg[j] for j in range(i)]
                if len(controls) == 1:
                    circ.cx(controls[0], reg[i])
                elif len(controls) == 2:
                    circ.ccx(controls[0], controls[1], reg[i])
                else:
                    circ.mcx(controls, reg[i])

    def run(self, round_num: int, shots: int = 1024) -> Tuple[int, dict]:
        """Run the algorithm."""
        circ, regs = self.build_circuit(round_num)

        transpiled = transpile(circ, basis_gates=['cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 'h',
                                                  's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'swap', 'ccx', 'cz', 'cp'])

        sim = Aer.get_backend('aer_simulator')
        # Use MPS for larger circuits
        if transpiled.num_qubits > 25:
            sim.set_options(method='matrix_product_state')
        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()

        best_s = max(counts.keys(), key=lambda x: counts[x])
        return int(best_s, 2), counts


def test_hybrid():
    """Test the hybrid implementation."""
    print("=" * 60)
    print("Hybrid Quantum Ladder Test")
    print("=" * 60)

    # y = 2x data
    data = [DataPoint(1, 2), DataPoint(2, 4)]
    k = 2

    print("\nDataset: y = 2x")
    print("Expected optimal: a* = 2")

    ladder = HybridQuantumLadder(data, coeff_bits=k)

    # Show expected winners
    bit_to_flip = 1  # Round 2
    print(f"\nExpected winners for round 2 (flip bit {bit_to_flip}):")
    for a in range(1 << k):
        a_prime = a ^ (1 << bit_to_flip)
        f_a = ladder.valuation_sums[a]
        f_ap = ladder.valuation_sums[a_prime]
        winner = a if f_a >= f_ap else a_prime
        print(f"  a={a} vs a'={a_prime}: F(a)={f_a}, F(a')={f_ap} → winner={winner}")

    # Run
    print("\n--- Running Hybrid Circuit ---")
    winner, counts = ladder.run(round_num=2, shots=1024)

    print(f"\nMeasurement results: {counts}")

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print("\nTop results:")
    for bitstring, count in sorted_counts[:4]:
        val = int(bitstring, 2)
        print(f"  s={val}: {count} shots ({100*count/1024:.1f}%)")

    if winner == 2:
        print("\n✓ Correct! Found optimal coefficient a* = 2")
    else:
        print(f"\n✗ Expected a* = 2, got {winner}")


def test_3bit():
    """Test with 3-bit coefficients."""
    print("\n" + "=" * 60)
    print("3-bit Hybrid Quantum Ladder Test")
    print("=" * 60)

    # y = 5x data
    data = [DataPoint(1, 5), DataPoint(2, 10)]
    k = 3

    print("\nDataset: y = 5x")
    print("Expected optimal: a* = 5")

    ladder = HybridQuantumLadder(data, coeff_bits=k)

    winner, counts = ladder.run(round_num=k, shots=4096)

    print(f"\nMeasurement results (top 5): {dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:3]
    print("\nTop 3 results:")
    for bitstring, count in sorted_counts:
        val = int(bitstring, 2)
        print(f"  s={val}: {count} shots ({100*count/4096:.1f}%)")

    if winner == 5:
        print("\n✓ Correct! Found optimal coefficient a* = 5")
    else:
        print(f"\n✗ Expected a* = 5, got {winner}")


if __name__ == "__main__":
    test_hybrid()
    test_3bit()
