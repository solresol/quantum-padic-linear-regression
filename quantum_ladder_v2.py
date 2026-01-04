#!/usr/bin/env python3
"""
Quantum Ladder Algorithm for 2-adic Linear Regression (Implementation Skeleton)

This implements the structure of the full quantum algorithm as described in
ALGORITHM_SUMMARY.md:

1. Prepare superposition over ALL possible coefficient values (2^k states)
2. For each round t:
   a. Compute F(a) = residual sum using coefficient a
   b. Compute F(a ⊕ 2^{t-1}) = residual sum with bit t-1 flipped
   c. Compare: apply phase flip if F(a) >= F(a ⊕ 2^{t-1})
   d. Uncompute F values
   e. Apply QFT to coefficient register
   f. Measure and extract bit t-1

The p-adic ladder structure should cause amplitudes to reinforce on the correct digit.

STATUS: This is a work-in-progress implementation. The circuit structure is correct
but there are likely bugs in the quantum arithmetic or uncomputation that prevent
the algorithm from converging to the correct answer. The measurements are currently
roughly uniform instead of concentrating on the optimal coefficient.

KNOWN ISSUES:
- Uncomputation may not be complete, leaving entanglement that interferes with QFT
- The trailing zeros counting (twoadic.py) may have subtle bugs in reversibility
- The comparison logic may have sign/direction errors
- Circuit is very large (~47 qubits) making debugging difficult

DEBUGGING NEEDED:
- Verify each component (multiply, subtract, trailing zeros, add) individually
- Check that uncomputation fully disentangles auxiliary registers
- Verify phase kickback is applied in correct direction
- Consider using statevector simulation on small cases to inspect state
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import math
from typing import List, Tuple

from padic_core import DataPoint, classical_valuation_sum, bits_needed
import quantum_arithmetic
import twoadic
import initialise


class QuantumLadderV2:
    """
    Correct implementation of the quantum ladder algorithm for p=2.

    Uses full superposition over 2^k coefficient values and quantum
    computation of residuals with proper uncomputation.
    """

    def __init__(self, data: List[DataPoint], coeff_bits: int = 8):
        """
        Initialize the quantum ladder algorithm.

        Args:
            data: List of (x, y) data points
            coeff_bits: Number of bits for coefficient (k in the algorithm)
        """
        self.data = data
        self.n_points = len(data)
        self.k = coeff_bits

        # Compute bounds for register sizes
        self.max_x = max(abs(pt.x) for pt in data)
        self.max_y = max(abs(pt.y) for pt in data)

        # Bits needed for residual computation
        # residual = y - a*x, max value is max_y + 2^k * max_x
        max_residual = self.max_y + (1 << self.k) * self.max_x
        self.residual_bits = min(bits_needed(max_residual) + 1, 6)  # cap for simulation

        # Bits for valuation count (max is residual_bits for all zeros)
        self.valuation_bits = min(bits_needed(self.residual_bits) + 1, 4)

        # Bits for valuation sum (max is n_points * max_valuation)
        self.sum_bits = min(bits_needed(self.n_points * self.residual_bits) + 1, 5)

    def build_round_circuit(self, t: int) -> Tuple[QuantumCircuit, dict]:
        """
        Build quantum circuit for round t.

        In round t, we find bit (t-1) of the optimal coefficient by:
        1. Superposition over all 2^k coefficient values
        2. For each |a⟩: compute F(a) and F(a ⊕ 2^{t-1})
        3. Compare and mark s=1 if F(a) is better
        4. Uncompute F values
        5. Apply QFT and measure

        Args:
            t: Round number (1 = find bit 0, 2 = find bit 1, etc.)

        Returns:
            (circuit, registers_dict)
        """
        # Main coefficient register - k qubits for values 0 to 2^k - 1
        a_reg = QuantumRegister(self.k, 'a')

        # Registers for computing F(a)
        # For each data point: product, residual, valuation
        # Then accumulate into sum
        product_reg = QuantumRegister(self.residual_bits, 'prod')
        residual_reg = QuantumRegister(self.residual_bits, 'resid')
        valuation_reg = QuantumRegister(self.valuation_bits, 'val')
        sum_a_reg = QuantumRegister(self.sum_bits, 'sum_a')

        # Registers for computing F(a ⊕ 2^{t-1})
        sum_a_flip_reg = QuantumRegister(self.sum_bits, 'sum_af')

        # Marker qubit: s=1 if F(a) ≤ F(a ⊕ 2^{t-1})
        marker_reg = QuantumRegister(1, 'marker')

        # Scratch registers for arithmetic (minimized for simulation)
        arith_scratch_size = max(self.residual_bits, self.sum_bits) + 1
        arith_scratch = QuantumRegister(arith_scratch_size, 'arith')

        # Scratch for trailing zero counting
        # Needs: 1 (still_zero) + 2 + valuation_bits (for increment) + 2 (extra for stop_if)
        tz_scratch_size = 1 + 2 + self.valuation_bits + 3
        tz_scratch = QuantumRegister(tz_scratch_size, 'tz')

        # Classical register for measurement
        a_classical = ClassicalRegister(self.k, 'a_c')

        # Build circuit
        circ = QuantumCircuit(
            a_reg, product_reg, residual_reg, valuation_reg,
            sum_a_reg, sum_a_flip_reg, marker_reg,
            arith_scratch, tz_scratch, a_classical
        )

        # Step 1: Prepare uniform superposition over all coefficient values
        for qubit in a_reg:
            circ.h(qubit)

        circ.barrier(label='superposition')

        # Step 2a: Compute F(a) - sum of valuations for coefficient a
        self._compute_residual_sum(circ, a_reg, sum_a_reg,
                                   product_reg, residual_reg, valuation_reg,
                                   arith_scratch, tz_scratch)

        circ.barrier(label='F(a) computed')

        # Step 2b: Flip bit (t-1) to get a' = a ⊕ 2^{t-1}
        bit_to_flip = t - 1
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # Step 2c: Compute F(a') = F(a ⊕ 2^{t-1})
        self._compute_residual_sum(circ, a_reg, sum_a_flip_reg,
                                   product_reg, residual_reg, valuation_reg,
                                   arith_scratch, tz_scratch)

        circ.barrier(label='F(a_flip) computed')

        # Step 2d: Flip bit back to restore a
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # Step 3: Compare F(a) vs F(a') and set marker
        # s = 1 if F(a) ≤ F(a'), i.e., if a is at least as good as a'
        # For 2-adic regression, HIGHER valuation sum is BETTER
        # So s = 1 if sum_a >= sum_a_flip
        self._compare_and_mark(circ, sum_a_reg, sum_a_flip_reg, marker_reg, arith_scratch)

        circ.barrier(label='comparison done')

        # Step 4: Uncompute F(a') and F(a)
        # First uncompute F(a')
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])
        self._uncompute_residual_sum(circ, a_reg, sum_a_flip_reg,
                                      product_reg, residual_reg, valuation_reg,
                                      arith_scratch, tz_scratch)
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # Then uncompute F(a)
        self._uncompute_residual_sum(circ, a_reg, sum_a_reg,
                                      product_reg, residual_reg, valuation_reg,
                                      arith_scratch, tz_scratch)

        circ.barrier(label='uncomputed')

        # Step 5: Apply QFT to coefficient register
        self._apply_qft(circ, a_reg)

        circ.barrier(label='QFT')

        # Step 6: Measure coefficient register
        circ.measure(a_reg, a_classical)

        registers = {
            'a': a_reg,
            'sum_a': sum_a_reg,
            'sum_a_flip': sum_a_flip_reg,
            'marker': marker_reg,
            'a_classical': a_classical
        }

        return circ, registers

    def _compute_residual_sum(self, circ, a_reg, sum_reg,
                               product_reg, residual_reg, valuation_reg,
                               arith_scratch, tz_scratch):
        """
        Compute F(a) = sum of 2-adic valuations of residuals.

        For each data point (x_i, y_i):
            residual_i = y_i - a * x_i
            valuation_i = v_2(residual_i) = trailing zeros
            sum += valuation_i
        """
        for i, pt in enumerate(self.data):
            # Reset working registers (should already be |0⟩)
            # In practice, they are reused after uncomputation

            # Step 1: Compute a * x_i into product_reg
            if pt.x != 0:
                quantum_arithmetic.multiply_by_constant(
                    circ, a_reg, pt.x, product_reg, arith_scratch
                )

            # Step 2: Compute y_i - (a * x_i) into residual_reg
            # residual = y_i - product
            # First initialize residual with y_i
            initialise.initialise_from_int(circ, residual_reg, pt.y)

            # Then subtract product (residual = y_i - a*x_i)
            if pt.x != 0:
                self._quantum_subtract_registers(circ, product_reg, residual_reg, arith_scratch)

            # Step 3: Count trailing zeros (2-adic valuation)
            twoadic.count_trailing_zeros_inplace(
                circ, residual_reg, valuation_reg, tz_scratch,
                name_prefix=f"ctz_{i}"
            )

            # Step 4: Add valuation to running sum
            quantum_arithmetic.quantum_add(
                circ, valuation_reg, sum_reg, arith_scratch
            )

            # Step 5: Uncompute valuation (run trailing zeros in reverse)
            self._uncompute_trailing_zeros(circ, residual_reg, valuation_reg, tz_scratch, f"unctz_{i}")

            # Step 6: Uncompute residual
            if pt.x != 0:
                self._quantum_add_registers(circ, product_reg, residual_reg, arith_scratch)
            initialise.initialise_from_int(circ, residual_reg, pt.y)  # XORs back to 0

            # Step 7: Uncompute product
            if pt.x != 0:
                self._uncompute_multiply(circ, a_reg, pt.x, product_reg, arith_scratch)

    def _uncompute_residual_sum(self, circ, a_reg, sum_reg,
                                 product_reg, residual_reg, valuation_reg,
                                 arith_scratch, tz_scratch):
        """
        Uncompute F(a) by running _compute_residual_sum in reverse.
        """
        # Process data points in reverse order
        for i in range(len(self.data) - 1, -1, -1):
            pt = self.data[i]

            # Redo computation up to the add-to-sum step, then reverse
            if pt.x != 0:
                quantum_arithmetic.multiply_by_constant(
                    circ, a_reg, pt.x, product_reg, arith_scratch
                )

            initialise.initialise_from_int(circ, residual_reg, pt.y)

            if pt.x != 0:
                self._quantum_subtract_registers(circ, product_reg, residual_reg, arith_scratch)

            twoadic.count_trailing_zeros_inplace(
                circ, residual_reg, valuation_reg, tz_scratch,
                name_prefix=f"rectz_{i}"
            )

            # SUBTRACT valuation from sum (reverse of add)
            self._quantum_subtract_registers(circ, valuation_reg, sum_reg, arith_scratch)

            # Uncompute valuation
            self._uncompute_trailing_zeros(circ, residual_reg, valuation_reg, tz_scratch, f"un2ctz_{i}")

            # Uncompute residual
            if pt.x != 0:
                self._quantum_add_registers(circ, product_reg, residual_reg, arith_scratch)
            initialise.initialise_from_int(circ, residual_reg, pt.y)

            # Uncompute product
            if pt.x != 0:
                self._uncompute_multiply(circ, a_reg, pt.x, product_reg, arith_scratch)

    def _quantum_subtract_registers(self, circ, a_reg, b_reg, scratch):
        """
        Compute b = b - a (in place).
        Uses two's complement: b - a = b + (~a + 1)
        """
        n = min(len(a_reg), len(b_reg))

        # Flip a bits (compute ~a)
        for i in range(n):
            circ.x(a_reg[i])

        # Add ~a to b
        quantum_arithmetic.quantum_add(circ, a_reg, b_reg, scratch)

        # Flip back
        for i in range(n):
            circ.x(a_reg[i])

        # Add 1 to complete two's complement
        quantum_arithmetic._add_one(circ, b_reg)

    def _quantum_add_registers(self, circ, a_reg, b_reg, scratch):
        """
        Compute b = b + a (in place).
        """
        quantum_arithmetic.quantum_add(circ, a_reg, b_reg, scratch)

    def _uncompute_multiply(self, circ, input_reg, constant, output_reg, scratch):
        """
        Uncompute multiplication by running it in reverse.
        Since multiply_by_constant uses controlled additions, we do controlled subtractions.
        """
        n_in = len(input_reg)

        # Run in reverse order
        for bit_pos in range(n_in - 1, -1, -1):
            shifted_constant = constant << bit_pos
            # Controlled subtract instead of add
            self._controlled_subtract_classical(circ, input_reg[bit_pos], output_reg,
                                                 shifted_constant, scratch)

    def _controlled_subtract_classical(self, circ, control, target_reg, value, scratch):
        """
        If control=1, compute target -= value.
        """
        n = len(target_reg)
        neg_value = (1 << n) - (value % (1 << n))
        quantum_arithmetic.controlled_add_classical(circ, control, target_reg, neg_value, scratch)

    def _uncompute_trailing_zeros(self, circ, diff_reg, tz_reg, anc_reg, name_prefix):
        """
        Uncompute trailing zero count by running the algorithm in reverse.

        This is complex because count_trailing_zeros_inplace modifies the
        still_zero ancilla. We need to carefully reverse all operations.
        """
        n = len(diff_reg)

        still_zero = anc_reg[0]
        scratch_for_increment = anc_reg[1 : 1 + 2 + len(tz_reg)]
        extra_scratch = anc_reg[1 + 2 + len(tz_reg):]

        # Run in reverse order
        for i in range(n - 1, -1, -1):
            # Reverse of stop_if_bit_is_1
            twoadic.stop_if_bit_is_1(
                circ, still_zero, diff_reg[i], extra_scratch[1:],
                name_prefix=f"{name_prefix}_unstop_bit{i}"
            )

            # Reverse the increment
            circ.x(diff_reg[i])
            all_controls = extra_scratch[0]
            circ.ccx(still_zero, diff_reg[i], all_controls)

            # Decrement (reverse of increment)
            twoadic.increment_by_one_if(
                circ, tz_reg, controls=[all_controls],
                carry_flag=scratch_for_increment[0],
                scratch=scratch_for_increment[1:],
                name_prefix=f"{name_prefix}_dec_bit{i}"
            )

            circ.ccx(still_zero, diff_reg[i], all_controls)
            circ.x(diff_reg[i])

        # Unset still_zero
        circ.x(still_zero)

    def _apply_qft(self, circ, qubits):
        """
        Apply Quantum Fourier Transform to the given qubits.
        Implements QFT with explicit gates (no library call).
        """
        n = len(qubits)

        # QFT implementation
        for i in range(n):
            circ.h(qubits[i])
            for j in range(i + 1, n):
                angle = math.pi / (1 << (j - i))
                circ.cp(angle, qubits[j], qubits[i])

        # Swap qubits to match standard QFT output ordering
        for i in range(n // 2):
            circ.swap(qubits[i], qubits[n - 1 - i])

    def _compare_and_mark(self, circ, sum_a_reg, sum_a_flip_reg, marker_reg, scratch):
        """
        Compare sum_a vs sum_a_flip and apply phase flip if sum_a >= sum_a_flip.

        For 2-adic regression, HIGHER sum is BETTER (more trailing zeros).
        If a is at least as good as a_flip, we apply a phase flip to that
        computational basis state. This is essential for the QFT to cause
        amplitude reinforcement.

        The phase flip is done via: compute comparison result into marker,
        apply Z gate to marker (phase flip if marker=1), then uncompute marker.
        """
        n = len(sum_a_reg)

        # Compute sum_a - sum_a_flip directly into sum_a
        self._quantum_subtract_registers(circ, sum_a_flip_reg, sum_a_reg, scratch)

        # Check sign: if MSB of sum_a is 0, result is non-negative (sum_a >= sum_a_flip)
        # Set marker = 1 if sum_a >= sum_a_flip (i.e., if MSB is 0)
        circ.x(sum_a_reg[n-1])  # Flip MSB so we can use it as control
        circ.cx(sum_a_reg[n-1], marker_reg[0])  # Copy to marker

        # Apply phase flip to states where marker=1 (i.e., where a is at least as good)
        circ.z(marker_reg[0])

        # Uncompute marker
        circ.cx(sum_a_reg[n-1], marker_reg[0])
        circ.x(sum_a_reg[n-1])  # Flip MSB back

        # Restore sum_a: add sum_a_flip back
        self._quantum_add_registers(circ, sum_a_flip_reg, sum_a_reg, scratch)

    def run_round(self, t: int, shots: int = 1024) -> Tuple[int, dict]:
        """
        Run round t of the algorithm.

        Returns:
            (extracted_bit_value, measurement_counts)
        """
        circ, registers = self.build_round_circuit(t)

        # Transpile circuit to decompose multi-controlled gates
        from qiskit import transpile
        transpiled = transpile(circ, basis_gates=['cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'swap', 'ccx', 'cz', 'cp'])

        # Run circuit using matrix product state simulator for large circuits
        sim = Aer.get_backend('aer_simulator')
        sim.set_options(method='matrix_product_state')
        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()

        # Find most common measurement
        best_count = 0
        best_a = 0

        for bitstring, count in counts.items():
            a_val = int(bitstring.replace(' ', ''), 2)
            if count > best_count:
                best_count = count
                best_a = a_val

        # Extract bit (t-1) from the measured value
        bit_value = (best_a >> (t - 1)) & 1

        return bit_value, counts

    def run_full_algorithm(self) -> int:
        """
        Run the full quantum ladder algorithm to find optimal coefficient.

        Returns:
            optimal_coefficient
        """
        print(f"Running quantum ladder algorithm with {self.k} bits")
        print(f"Data points: {self.n_points}")

        result = 0

        for t in range(1, self.k + 1):
            print(f"\n--- Round {t}: Finding bit {t-1} ---")

            bit_value, counts = self.run_round(t)

            result |= (bit_value << (t - 1))

            print(f"Extracted bit {t-1} = {bit_value}")
            print(f"Current result: {result} (binary: {bin(result)})")

            # Show top measurement counts
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:3]
            print(f"Top measurements: {sorted_counts}")

        print(f"\n=== Final result: a = {result} ===")
        print(f"Valuation sum: {classical_valuation_sum(self.data, result, 0)}")

        return result


def verify_classical_valuations():
    """Verify that the classical valuation computation is correct."""
    print("=" * 60)
    print("Verifying Classical Valuations")
    print("=" * 60)

    data = [DataPoint(1, 2), DataPoint(2, 4)]

    print("\nDataset: y = 2x")
    for a in range(8):
        val_sum = classical_valuation_sum(data, a, 0)
        residuals = [pt.y - a * pt.x for pt in data]
        print(f"  a={a}: residuals={residuals}, valuation_sum={val_sum}")

    print("\nExpected: a=2 should have highest valuation (residuals all 0)")


def test_quantum_ladder_v2():
    """Test the quantum ladder algorithm."""
    print("=" * 60)
    print("Quantum Ladder V2 Algorithm Test")
    print("=" * 60)

    # First verify classical computations
    verify_classical_valuations()

    # Simple test: y = 2x, so optimal coefficient is 2
    data = [
        DataPoint(1, 2),
        DataPoint(2, 4),
    ]

    print("\n--- Running Quantum Ladder V2 ---")
    # Use only 3 coefficient bits (values 0-7) to keep circuit small
    ladder = QuantumLadderV2(data, coeff_bits=3)

    # Print qubit count for debugging
    print(f"Register sizes: k={ladder.k}, residual={ladder.residual_bits}, "
          f"val={ladder.valuation_bits}, sum={ladder.sum_bits}")

    # Count total qubits
    total_qubits = (ladder.k + ladder.residual_bits * 2 + ladder.valuation_bits +
                    ladder.sum_bits * 2 + 1 +
                    max(ladder.residual_bits, ladder.sum_bits) + 1 +
                    1 + 2 + ladder.valuation_bits + 3)
    print(f"Estimated total qubits: {total_qubits}")

    result = ladder.run_full_algorithm()

    expected = 2
    if result == expected:
        print(f"\n✓ Correct! Found a = {result}")
    else:
        print(f"\n✗ Expected a = {expected}, got a = {result}")

    return result == expected


if __name__ == "__main__":
    test_quantum_ladder_v2()
