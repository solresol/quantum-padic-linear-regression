#!/usr/bin/env python3
"""
Quantum Oracle for p-adic Linear Regression

This module implements the quantum oracle that computes residual valuations
for the p-adic regression algorithm.

The oracle marks states (m, b) where the total 2-adic valuation sum
exceeds a threshold, enabling Grover-based search for optimal solutions.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from typing import List, Tuple
import math

import initialise
import twoadic
import quantum_arithmetic
from padic_core import DataPoint, classical_valuation_sum, bits_needed


class QuantumResidualOracle:
    """
    Quantum oracle for computing residuals and their 2-adic valuations.

    For a dataset of points (x_i, y_i) and candidate line parameters (m, b),
    computes the sum of 2-adic valuations of residuals y_i - (m*x_i + b).
    """

    def __init__(self, data: List[DataPoint], m_bits: int, b_bits: int):
        """
        Initialize the oracle.

        Args:
            data: List of data points
            m_bits: Number of bits for gradient m
            b_bits: Number of bits for intercept b
        """
        self.data = data
        self.m_bits = m_bits
        self.b_bits = b_bits
        self.n_points = len(data)

        # Compute sizes
        self.max_x = max(abs(p.x) for p in data)
        self.max_y = max(abs(p.y) for p in data)
        self.max_m = 2**m_bits - 1
        self.max_b = 2**b_bits - 1

        # Max residual magnitude
        self.max_residual = self.max_y + self.max_m * self.max_x + self.max_b
        self.residual_bits = bits_needed(self.max_residual) + 1  # +1 for potential overflow

        # Valuation bits (max valuation per point is residual_bits)
        self.val_bits = bits_needed(self.residual_bits)

        # Total valuation sum bits
        self.total_val_bits = bits_needed(self.n_points * self.residual_bits)

    def build_single_residual_circuit(self, x_val: int, y_val: int) -> Tuple[QuantumCircuit, dict]:
        """
        Build circuit to compute residual = y - (m*x + b) and its trailing zeros.

        Args:
            x_val: Classical x value for this data point
            y_val: Classical y value for this data point

        Returns:
            (circuit, register_dict) where register_dict maps names to registers
        """
        # Registers
        m_reg = QuantumRegister(self.m_bits, 'm')
        b_reg = QuantumRegister(self.b_bits, 'b')
        mx_reg = QuantumRegister(self.residual_bits, 'mx')  # m * x
        mxb_reg = QuantumRegister(self.residual_bits, 'mxb')  # m * x + b
        residual_reg = QuantumRegister(self.residual_bits, 'residual')  # y - (mx + b)
        tz_reg = QuantumRegister(self.val_bits, 'tz')  # trailing zeros count

        # Scratch registers
        mult_scratch = QuantumRegister(self.residual_bits, 'mult_scratch')
        add_scratch = QuantumRegister(self.residual_bits, 'add_scratch')
        sub_scratch = QuantumRegister(self.residual_bits, 'sub_scratch')
        tz_anc = QuantumRegister(1 + 2 + self.val_bits + 2, 'tz_anc')

        circ = QuantumCircuit(m_reg, b_reg, mx_reg, mxb_reg, residual_reg,
                              tz_reg, mult_scratch, add_scratch, sub_scratch, tz_anc)

        # Step 1: Compute m * x (x is classical constant)
        quantum_arithmetic.multiply_by_constant(circ, m_reg, x_val, mx_reg, mult_scratch)

        # Step 2: Compute m*x + b
        # First copy mx to mxb, then add b
        for i in range(min(self.residual_bits, len(mx_reg))):
            circ.cx(mx_reg[i], mxb_reg[i])

        # Add b to mxb
        quantum_arithmetic.quantum_add(circ, b_reg, mxb_reg, add_scratch)

        # Step 3: Compute y - (m*x + b)
        # Create a temporary register for y
        y_reg = QuantumRegister(self.residual_bits, 'y')
        circ.add_register(y_reg)
        initialise.initialise_from_int(circ, y_reg, y_val)

        quantum_arithmetic.quantum_subtract(circ, y_reg, mxb_reg, residual_reg, sub_scratch)

        # Step 4: Compute trailing zeros of residual
        twoadic.count_trailing_zeros_inplace(circ, residual_reg, tz_reg, tz_anc)

        registers = {
            'm': m_reg,
            'b': b_reg,
            'mx': mx_reg,
            'mxb': mxb_reg,
            'residual': residual_reg,
            'tz': tz_reg,
            'y': y_reg
        }

        return circ, registers

    def build_full_oracle_circuit(self) -> Tuple[QuantumCircuit, dict]:
        """
        Build the complete oracle circuit for all data points.

        This computes the total valuation sum for parameters (m, b).

        Returns:
            (circuit, register_dict)
        """
        # Main parameter registers
        m_reg = QuantumRegister(self.m_bits, 'm')
        b_reg = QuantumRegister(self.b_bits, 'b')

        # Total valuation accumulator
        total_val = QuantumRegister(self.total_val_bits, 'total_val')

        # Working registers (reused for each data point)
        mx_reg = QuantumRegister(self.residual_bits, 'mx')
        mxb_reg = QuantumRegister(self.residual_bits, 'mxb')
        residual_reg = QuantumRegister(self.residual_bits, 'residual')
        tz_reg = QuantumRegister(self.val_bits, 'tz')

        # Scratch
        scratch_size = max(self.residual_bits, 1 + 2 + self.val_bits + 2, self.total_val_bits)
        scratch = QuantumRegister(scratch_size, 'scratch')

        circ = QuantumCircuit(m_reg, b_reg, total_val, mx_reg, mxb_reg,
                              residual_reg, tz_reg, scratch)

        # Process each data point
        for point in self.data:
            self._add_point_valuation(circ, point, m_reg, b_reg,
                                       mx_reg, mxb_reg, residual_reg, tz_reg,
                                       total_val, scratch)

            # Reset working registers for next point (except total_val)
            self._reset_register(circ, mx_reg)
            self._reset_register(circ, mxb_reg)
            self._reset_register(circ, residual_reg)
            self._reset_register(circ, tz_reg)

        registers = {
            'm': m_reg,
            'b': b_reg,
            'total_val': total_val
        }

        return circ, registers

    def _add_point_valuation(self, circ, point, m_reg, b_reg,
                              mx_reg, mxb_reg, residual_reg, tz_reg,
                              total_val, scratch):
        """Add the valuation for one data point to total_val."""

        # Compute m * x
        quantum_arithmetic.multiply_by_constant(circ, m_reg, point.x, mx_reg, scratch)

        # Compute m*x + b into mxb_reg
        for i in range(len(mx_reg)):
            circ.cx(mx_reg[i], mxb_reg[i])
        quantum_arithmetic.quantum_add(circ, b_reg, mxb_reg, scratch)

        # Compute y - (m*x + b)
        # Need to load y into a register first
        y_bits = bits_needed(point.y) + 1
        y_temp = scratch[:y_bits]  # Borrow from scratch

        # Initialize y
        for i in range(min(y_bits, len(scratch))):
            if point.y & (1 << i):
                circ.x(scratch[i])

        # Subtract - this is complex because we're borrowing scratch
        # For simplicity, use a direct approach
        self._compute_residual_direct(circ, point.y, mxb_reg, residual_reg)

        # Clear y from scratch
        for i in range(min(y_bits, len(scratch))):
            if point.y & (1 << i):
                circ.x(scratch[i])

        # Compute trailing zeros
        tz_anc_size = 1 + 2 + len(tz_reg) + 2
        tz_anc = scratch[:tz_anc_size]
        twoadic.count_trailing_zeros_inplace(circ, residual_reg, tz_reg, tz_anc)

        # Add tz to total_val
        quantum_arithmetic.quantum_add(circ, tz_reg, total_val, scratch)

    def _compute_residual_direct(self, circ, y_val, mxb_reg, result_reg):
        """Compute result = y_val - mxb where y is a classical constant."""
        n = len(result_reg)

        # result = y - mxb = y + (~mxb + 1)

        # First, put y into result
        for i in range(n):
            if y_val & (1 << i):
                circ.x(result_reg[i])

        # Now add ~mxb + 1 (two's complement of mxb)
        # ~mxb[i] means: add 2^i if mxb[i] = 0
        for i in range(len(mxb_reg)):
            circ.x(mxb_reg[i])
            quantum_arithmetic._controlled_add_power_of_2(circ, mxb_reg[i], result_reg, i, None)
            circ.x(mxb_reg[i])

        # Add 1
        quantum_arithmetic._add_one(circ, result_reg)

    def _reset_register(self, circ, reg):
        """Reset a register to |0> (assumes it's in a computational basis state)."""
        # In a real implementation, we'd uncompute. For now, just add resets.
        # Note: reset is not unitary, so this breaks reversibility
        # For a proper implementation, we'd need to uncompute the values
        for i in range(len(reg)):
            circ.reset(reg[i])


def build_grover_oracle(data: List[DataPoint], m_bits: int, b_bits: int,
                        threshold: int) -> QuantumCircuit:
    """
    Build a Grover oracle that marks (m, b) pairs with valuation sum >= threshold.

    Args:
        data: Data points
        m_bits: Bits for gradient
        b_bits: Bits for intercept
        threshold: Minimum valuation sum to mark

    Returns:
        Oracle circuit
    """
    oracle_builder = QuantumResidualOracle(data, m_bits, b_bits)

    m_reg = QuantumRegister(m_bits, 'm')
    b_reg = QuantumRegister(b_bits, 'b')

    # For a simple oracle, we compute the valuation sum classically
    # and encode it into the circuit

    circ = QuantumCircuit(m_reg, b_reg)

    # Mark each (m, b) that meets threshold
    for m in range(2**m_bits):
        for b in range(2**b_bits):
            val_sum = classical_valuation_sum(data, m, b)
            if val_sum >= threshold:
                # Apply Z to this state
                _mark_state(circ, m_reg, b_reg, m, b)

    return circ


def _mark_state(circ, m_reg, b_reg, m_val, b_val):
    """Apply phase flip to state |m_val, b_val>."""
    m_bits = len(m_reg)
    b_bits = len(b_reg)

    # Flip bits that should be 0
    for i in range(m_bits):
        if not (m_val & (1 << i)):
            circ.x(m_reg[i])
    for i in range(b_bits):
        if not (b_val & (1 << i)):
            circ.x(b_reg[i])

    # Multi-controlled Z
    all_qubits = list(m_reg) + list(b_reg)
    if len(all_qubits) == 1:
        circ.z(all_qubits[0])
    elif len(all_qubits) == 2:
        circ.cz(all_qubits[0], all_qubits[1])
    else:
        # MCZ = H on target, MCX, H on target
        target = all_qubits[-1]
        controls = all_qubits[:-1]
        circ.h(target)
        circ.mcx(controls, target)
        circ.h(target)

    # Unflip
    for i in range(m_bits):
        if not (m_val & (1 << i)):
            circ.x(m_reg[i])
    for i in range(b_bits):
        if not (b_val & (1 << i)):
            circ.x(b_reg[i])


def grover_diffusion(circ, m_reg, b_reg):
    """Apply Grover diffusion operator."""
    all_qubits = list(m_reg) + list(b_reg)

    # H on all
    for q in all_qubits:
        circ.h(q)

    # X on all
    for q in all_qubits:
        circ.x(q)

    # Multi-controlled Z
    if len(all_qubits) == 1:
        circ.z(all_qubits[0])
    elif len(all_qubits) == 2:
        circ.cz(all_qubits[0], all_qubits[1])
    else:
        target = all_qubits[-1]
        controls = all_qubits[:-1]
        circ.h(target)
        circ.mcx(controls, target)
        circ.h(target)

    # X on all
    for q in all_qubits:
        circ.x(q)

    # H on all
    for q in all_qubits:
        circ.h(q)


def quantum_find_optimal(data: List[DataPoint], m_bits: int, b_bits: int) -> Tuple[int, int, dict]:
    """
    Use Grover's algorithm to find optimal (m, b) for p-adic regression.

    Uses binary search on the threshold to find the maximum valuation sum,
    then Grover search to find an (m, b) achieving it.

    Args:
        data: Data points
        m_bits: Bits for gradient
        b_bits: Bits for intercept

    Returns:
        (optimal_m, optimal_b, measurement_counts)
    """
    # First, find the optimal threshold via classical search
    # (In a truly quantum version, we'd use quantum minimum finding)
    best_val = 0
    for m in range(2**m_bits):
        for b in range(2**b_bits):
            val = classical_valuation_sum(data, m, b)
            best_val = max(best_val, val)

    # Build Grover circuit to find (m, b) with valuation >= best_val
    m_reg = QuantumRegister(m_bits, 'm')
    b_reg = QuantumRegister(b_bits, 'b')
    m_c = ClassicalRegister(m_bits, 'm_c')
    b_c = ClassicalRegister(b_bits, 'b_c')

    circ = QuantumCircuit(m_reg, b_reg, m_c, b_c)

    # Initial superposition
    circ.h(m_reg)
    circ.h(b_reg)

    # Count solutions to determine iterations
    n_solutions = sum(1 for m in range(2**m_bits) for b in range(2**b_bits)
                      if classical_valuation_sum(data, m, b) >= best_val)
    n_total = 2**(m_bits + b_bits)

    if n_solutions == 0:
        # No solutions, just measure
        pass
    elif n_solutions == n_total:
        # All solutions work, just measure
        pass
    else:
        # Compute optimal number of Grover iterations
        theta = math.asin(math.sqrt(n_solutions / n_total))
        if theta > 0:
            n_iterations = max(1, int(round(math.pi / (4 * theta) - 0.5)))
        else:
            n_iterations = 1

        # Apply Grover iterations
        for _ in range(n_iterations):
            # Oracle
            oracle = build_grover_oracle(data, m_bits, b_bits, best_val)
            circ.compose(oracle, qubits=list(m_reg) + list(b_reg), inplace=True)

            # Diffusion
            grover_diffusion(circ, m_reg, b_reg)

    # Measure
    circ.measure(m_reg, m_c)
    circ.measure(b_reg, b_c)

    # Run
    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=1024).result()
    counts = result.get_counts()

    # Find most common result
    best_count = 0
    result_m, result_b = 0, 0
    for bitstring, count in counts.items():
        if count > best_count:
            best_count = count
            parts = bitstring.replace(' ', '')
            # Qiskit bit ordering: rightmost is qubit 0
            b_str = parts[:b_bits]
            m_str = parts[b_bits:]
            result_b = int(b_str, 2)
            result_m = int(m_str, 2)

    return result_m, result_b, counts


# =============================================================================
# Testing
# =============================================================================

def test_grover_search():
    """Test the Grover-based optimal search."""
    print("Testing quantum_find_optimal...")

    # Test case 1: Points on y = 2x + 1
    data = [
        DataPoint(0, 1),
        DataPoint(1, 3),
        DataPoint(2, 5),
        DataPoint(3, 7),
    ]

    print("\nDataset: y = 2x + 1")
    for p in data:
        print(f"  ({p.x}, {p.y})")

    m, b, counts = quantum_find_optimal(data, m_bits=3, b_bits=3)
    print(f"\nQuantum result: m={m}, b={b}")
    print(f"Valuation sum: {classical_valuation_sum(data, m, b)}")
    print(f"Top measurement counts: {dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")

    # Verify
    expected_m, expected_b = 2, 1
    if m == expected_m and b == expected_b:
        print("✓ Correct!")
    else:
        print(f"✗ Expected m={expected_m}, b={expected_b}")

    # Test case 2: Different dataset
    print("\n" + "="*50)
    data2 = [
        DataPoint(0, 0),
        DataPoint(1, 2),
        DataPoint(2, 4),
        DataPoint(3, 8),
    ]

    print("\nDataset 2:")
    for p in data2:
        print(f"  ({p.x}, {p.y})")

    m, b, counts = quantum_find_optimal(data2, m_bits=3, b_bits=3)
    print(f"\nQuantum result: m={m}, b={b}")
    print(f"Valuation sum: {classical_valuation_sum(data2, m, b)}")

    expected_m, expected_b = 2, 0
    if m == expected_m and b == expected_b:
        print("✓ Correct!")
    else:
        print(f"✗ Expected m={expected_m}, b={expected_b}")


if __name__ == "__main__":
    test_grover_search()
