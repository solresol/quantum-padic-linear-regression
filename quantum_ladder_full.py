#!/usr/bin/env python3
"""
Fully Quantum Ladder Algorithm for 2-adic Linear Regression

This implementation uses quantum arithmetic throughout - no classical
precomputation of winners. The algorithm:

1. Prepare superposition over all coefficient values
2. For each |a⟩ in superposition:
   - Quantumly compute F(a) = sum of 2-adic valuations
   - Compute a' = a ⊕ 2^{t-1} (flip bit t-1)
   - Quantumly compute F(a')
   - Compare F(a) vs F(a')
   - Copy winner value to s register
   - Uncompute F(a) and F(a')
3. Measure s to get the optimal coefficient

Key insight: States with the same winner have probabilities that add,
making the optimal coefficient the most likely measurement outcome.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit import transpile
import math
from typing import List, Tuple

from padic_core import DataPoint, classical_valuation_sum
import quantum_arithmetic
import twoadic
import initialise


class FullQuantumLadder:
    """
    Fully quantum implementation of the p-adic ladder algorithm for p=2.
    """

    def __init__(self, data: List[DataPoint], coeff_bits: int = 2):
        """
        Initialize the quantum ladder algorithm.

        Args:
            data: List of (x, y) data points (should be y = a*x format)
            coeff_bits: Number of bits for coefficient (k)
        """
        self.data = data
        self.n_points = len(data)
        self.k = coeff_bits

        # Compute register sizes
        self.max_x = max(abs(pt.x) for pt in data)
        self.max_y = max(abs(pt.y) for pt in data)

        # Bits for residual y - a*x
        max_product = (1 << self.k) * self.max_x
        max_residual = max(self.max_y, max_product) * 2
        self.residual_bits = max(4, initialise.number_of_bits_required(max_residual) + 1)

        # Bits for product a*x: MUST match residual_bits for subtraction to work
        # (two's complement subtraction requires same-sized registers)
        self.product_bits = self.residual_bits

        # Bits for valuation (trailing zeros count): max is residual_bits
        # Cap at residual_bits since that's the max meaningful valuation
        self.valuation_bits = max(3, initialise.number_of_bits_required(self.residual_bits) + 1)

        # Bits for valuation sum: max is n_points * max_valuation
        # The quantum trailing zeros counter returns at most residual_bits
        # (unlike the classical function which returns 1000 for zero)
        max_valuation_per_point = self.residual_bits  # Max trailing zeros
        max_sum = self.n_points * max_valuation_per_point
        self.sum_bits = max(4, initialise.number_of_bits_required(max_sum) + 1)

        print(f"Register sizes: k={self.k}, product={self.product_bits}, "
              f"residual={self.residual_bits}, valuation={self.valuation_bits}, sum={self.sum_bits}")

    def build_circuit(self, round_num: int = None) -> Tuple[QuantumCircuit, dict]:
        """
        Build the full quantum circuit.

        Args:
            round_num: Which round (determines bit to flip). If None, compares
                      full coefficient values (not just one bit flip).

        Returns:
            (circuit, registers_dict)
        """
        # Main coefficient register
        a_reg = QuantumRegister(self.k, 'a')

        # Winner register (stores the winning coefficient)
        s_reg = QuantumRegister(self.k, 's')

        # Working registers for residual computation
        product_reg = QuantumRegister(self.product_bits, 'prod')
        residual_reg = QuantumRegister(self.residual_bits, 'resid')
        valuation_reg = QuantumRegister(self.valuation_bits, 'val')

        # Sum registers for F(a) and F(a')
        sum_a_reg = QuantumRegister(self.sum_bits, 'sum_a')
        sum_a_prime_reg = QuantumRegister(self.sum_bits, 'sum_ap')

        # Comparison result register
        cmp_reg = QuantumRegister(1, 'cmp')

        # Scratch registers
        arith_scratch_size = max(self.product_bits, self.residual_bits, self.sum_bits) + 2
        arith_scratch = QuantumRegister(arith_scratch_size, 'arith')

        # Scratch for trailing zeros:
        # 1 (still_zero) + 2 + valuation_bits (increment) + 1 (all_controls) + residual_bits (stop tmps)
        tz_scratch_size = 1 + 2 + self.valuation_bits + 1 + self.residual_bits
        tz_scratch = QuantumRegister(tz_scratch_size, 'tz')

        # Classical register for measurement
        s_classical = ClassicalRegister(self.k, 's_c')

        # Build circuit
        circ = QuantumCircuit(
            a_reg, s_reg,
            product_reg, residual_reg, valuation_reg,
            sum_a_reg, sum_a_prime_reg, cmp_reg,
            arith_scratch, tz_scratch,
            s_classical
        )

        # ===== Step 1: Superposition over all coefficient values =====
        for qubit in a_reg:
            circ.h(qubit)

        circ.barrier(label='superposition')

        # ===== Step 2: Compute F(a) = sum of valuations =====
        self._compute_residual_sum(circ, a_reg, sum_a_reg,
                                   product_reg, residual_reg, valuation_reg,
                                   arith_scratch, tz_scratch)

        circ.barrier(label='F(a)')

        # ===== Step 3: Flip bit to get a' = a ⊕ 2^{t-1} =====
        bit_to_flip = (round_num - 1) if round_num else (self.k - 1)
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # ===== Step 4: Compute F(a') =====
        self._compute_residual_sum(circ, a_reg, sum_a_prime_reg,
                                   product_reg, residual_reg, valuation_reg,
                                   arith_scratch, tz_scratch)

        circ.barrier(label="F(a')")

        # ===== Step 5: Flip bit back to restore a =====
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # ===== Step 6: Compare F(a) >= F(a') and set cmp =====
        # cmp = 1 if F(a) >= F(a') (meaning a is at least as good)
        self._compare_sums(circ, sum_a_reg, sum_a_prime_reg, cmp_reg, arith_scratch)

        circ.barrier(label='compare')

        # ===== Step 7: Copy winner to s register =====
        # If cmp=1 (a wins): s = a
        # If cmp=0 (a' wins): s = a ⊕ 2^{bit_to_flip}
        self._copy_winner(circ, a_reg, s_reg, cmp_reg, bit_to_flip)

        circ.barrier(label='copy_winner')

        # ===== Step 8: Uncompute comparison result =====
        self._compare_sums(circ, sum_a_reg, sum_a_prime_reg, cmp_reg, arith_scratch)

        # ===== Step 9: Uncompute F(a') =====
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])
        self._uncompute_residual_sum(circ, a_reg, sum_a_prime_reg,
                                      product_reg, residual_reg, valuation_reg,
                                      arith_scratch, tz_scratch)
        if bit_to_flip < self.k:
            circ.x(a_reg[bit_to_flip])

        # ===== Step 10: Uncompute F(a) =====
        self._uncompute_residual_sum(circ, a_reg, sum_a_reg,
                                      product_reg, residual_reg, valuation_reg,
                                      arith_scratch, tz_scratch)

        circ.barrier(label='uncomputed')

        # ===== Step 11: Measure s =====
        circ.measure(s_reg, s_classical)

        registers = {
            'a': a_reg,
            's': s_reg,
            'sum_a': sum_a_reg,
            'sum_a_prime': sum_a_prime_reg,
            'cmp': cmp_reg,
            's_classical': s_classical
        }

        return circ, registers

    def _compute_residual_sum(self, circ, a_reg, sum_reg,
                               product_reg, residual_reg, valuation_reg,
                               arith_scratch, tz_scratch):
        """
        Compute F(a) = sum of 2-adic valuations of residuals.

        For each data point (x_i, y_i):
            product = a * x_i
            residual = y_i - product
            valuation = trailing_zeros(residual)
            sum += valuation

        Working registers are reused via uncomputation after each point.
        """
        for i, pt in enumerate(self.data):
            # Step 1: Compute a * x_i into product_reg
            if pt.x != 0:
                quantum_arithmetic.multiply_by_constant(
                    circ, a_reg, pt.x, product_reg, arith_scratch
                )

            # Step 2: Initialize residual with y_i
            initialise.initialise_from_int(circ, residual_reg, pt.y)

            # Step 3: Compute residual = y_i - a*x_i
            if pt.x != 0:
                self._quantum_subtract_inplace(circ, product_reg, residual_reg, arith_scratch)

            # Step 4: Count trailing zeros (2-adic valuation)
            twoadic.count_trailing_zeros_inplace(
                circ, residual_reg, valuation_reg, tz_scratch,
                name_prefix=f"ctz_{i}"
            )

            # Step 5: Add valuation to running sum
            quantum_arithmetic.quantum_add(circ, valuation_reg, sum_reg, arith_scratch)

            # Step 6: Uncompute valuation
            self._uncompute_trailing_zeros(circ, residual_reg, valuation_reg, tz_scratch, f"unctz_{i}")

            # Step 7: Uncompute residual
            if pt.x != 0:
                self._quantum_add_inplace(circ, product_reg, residual_reg, arith_scratch)
            initialise.initialise_from_int(circ, residual_reg, pt.y)  # XOR back to 0

            # Step 8: Uncompute product
            if pt.x != 0:
                self._uncompute_multiply(circ, a_reg, pt.x, product_reg, arith_scratch)

    def _uncompute_residual_sum(self, circ, a_reg, sum_reg,
                                 product_reg, residual_reg, valuation_reg,
                                 arith_scratch, tz_scratch):
        """
        Uncompute F(a) by processing data points in reverse order
        and subtracting valuations instead of adding.
        """
        for i in range(len(self.data) - 1, -1, -1):
            pt = self.data[i]

            # Recompute up to the sum step
            if pt.x != 0:
                quantum_arithmetic.multiply_by_constant(
                    circ, a_reg, pt.x, product_reg, arith_scratch
                )

            initialise.initialise_from_int(circ, residual_reg, pt.y)

            if pt.x != 0:
                self._quantum_subtract_inplace(circ, product_reg, residual_reg, arith_scratch)

            twoadic.count_trailing_zeros_inplace(
                circ, residual_reg, valuation_reg, tz_scratch,
                name_prefix=f"rectz_{i}"
            )

            # SUBTRACT from sum (reverse of add)
            self._quantum_subtract_inplace(circ, valuation_reg, sum_reg, arith_scratch)

            # Uncompute valuation
            self._uncompute_trailing_zeros(circ, residual_reg, valuation_reg, tz_scratch, f"un2ctz_{i}")

            # Uncompute residual
            if pt.x != 0:
                self._quantum_add_inplace(circ, product_reg, residual_reg, arith_scratch)
            initialise.initialise_from_int(circ, residual_reg, pt.y)

            # Uncompute product
            if pt.x != 0:
                self._uncompute_multiply(circ, a_reg, pt.x, product_reg, arith_scratch)

    def _quantum_subtract_inplace(self, circ, a_reg, b_reg, scratch):
        """
        Compute b = b - a (in place) using two's complement.

        Uses a corrected in-place addition that computes carries before sums.

        When a_reg is shorter than b_reg, we need to sign-extend ~a.
        For ~a, the bits beyond len(a_reg) are all 1s (since a has leading 0s).
        """
        n_a = len(a_reg)
        n_b = len(b_reg)
        n = min(n_a, n_b)

        # Flip a bits (compute ~a for lower bits)
        for i in range(n):
            circ.x(a_reg[i])

        # For sign extension: if a_reg is shorter, ~a has 1s in the upper bits
        # We achieve this by flipping the upper bits of b_reg before and after addition
        # This effectively adds (2^n_b - 2^n_a) = 11...100...0 pattern
        for i in range(n, n_b):
            circ.x(b_reg[i])

        # Add ~a to b using corrected in-place addition
        self._quantum_add_inplace_correct(circ, a_reg, b_reg, scratch)

        # Flip back the sign extension bits
        for i in range(n, n_b):
            circ.x(b_reg[i])

        # Flip back a bits
        for i in range(n):
            circ.x(a_reg[i])

        # Add 1 to complete two's complement
        quantum_arithmetic._add_one(circ, b_reg)

    def _quantum_add_inplace_correct(self, circ, a_reg, b_reg, scratch):
        """
        Correct in-place addition: b = b + a.

        This uses the ripple-carry approach but uncomputes carries correctly.
        The key insight is that after computing sum b' = a + b:
        - To uncompute scratch, we need the ORIGINAL b values
        - Since b has been modified, we use: original_b = b' XOR a
        - So we XOR with a before using b for uncomputation, then XOR back
        """
        n = min(len(a_reg), len(b_reg))

        if len(scratch) < n:
            raise ValueError(f"Need at least {n} scratch qubits")

        # Phase 1: Compute all carries using original values
        circ.ccx(a_reg[0], b_reg[0], scratch[0])

        for i in range(1, n - 1):
            circ.ccx(a_reg[i], b_reg[i], scratch[i])
            circ.ccx(a_reg[i], scratch[i-1], scratch[i])
            circ.ccx(b_reg[i], scratch[i-1], scratch[i])

        # Phase 2: Compute sums from MSB down (so carries are still valid)
        for i in range(n - 1, 0, -1):
            circ.cx(a_reg[i], b_reg[i])
            circ.cx(scratch[i-1], b_reg[i])

        circ.cx(a_reg[0], b_reg[0])

        # Phase 3: Uncompute carries
        # We need original b values, but b now contains b+a
        # Temporarily restore b to original for uncomputation

        # First, restore b[0] to original by XORing with a[0]
        circ.cx(a_reg[0], b_reg[0])
        # Now uncompute scratch[0]
        circ.ccx(a_reg[0], b_reg[0], scratch[0])
        # Restore b[0] to sum
        circ.cx(a_reg[0], b_reg[0])

        # For higher bits, restore b[i] considering carry propagation
        for i in range(1, n - 1):
            # Restore b[i] to original
            circ.cx(a_reg[i], b_reg[i])
            circ.cx(scratch[i-1], b_reg[i])

            # Uncompute scratch[i] using original values
            circ.ccx(b_reg[i], scratch[i-1], scratch[i])
            circ.ccx(a_reg[i], scratch[i-1], scratch[i])
            circ.ccx(a_reg[i], b_reg[i], scratch[i])

            # Restore b[i] to sum
            circ.cx(scratch[i-1], b_reg[i])
            circ.cx(a_reg[i], b_reg[i])

    def _quantum_add_inplace(self, circ, a_reg, b_reg, scratch):
        """Compute b = b + a (in place)."""
        self._quantum_add_inplace_correct(circ, a_reg, b_reg, scratch)

    def _uncompute_multiply(self, circ, input_reg, constant, output_reg, scratch):
        """Uncompute multiplication by running controlled subtractions in reverse."""
        n_in = len(input_reg)
        for bit_pos in range(n_in - 1, -1, -1):
            shifted_constant = constant << bit_pos
            self._controlled_subtract_classical(circ, input_reg[bit_pos], output_reg,
                                                 shifted_constant, scratch)

    def _controlled_subtract_classical(self, circ, control, target_reg, value, scratch):
        """If control=1, compute target -= value."""
        n = len(target_reg)
        # Two's complement: subtract value by adding its negative
        neg_value = (1 << n) - (value % (1 << n))
        quantum_arithmetic.controlled_add_classical(circ, control, target_reg, neg_value, scratch)

    def _uncompute_trailing_zeros(self, circ, diff_reg, tz_reg, anc_reg, name_prefix):
        """
        Uncompute trailing zero count by running in reverse.
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

    def _compare_sums(self, circ, sum_a_reg, sum_a_prime_reg, cmp_reg, scratch):
        """
        Compare sum_a >= sum_a_prime and set cmp=1 if true.

        Uses subtraction: compute sum_a - sum_a_prime, check sign bit.
        Higher valuation sum is better, so we want cmp=1 when sum_a >= sum_a_prime.
        """
        n = len(sum_a_reg)

        # Compute sum_a - sum_a_prime into sum_a (destructive)
        self._quantum_subtract_inplace(circ, sum_a_prime_reg, sum_a_reg, scratch)

        # If MSB is 0, result is non-negative (sum_a >= sum_a_prime)
        # Set cmp = NOT(MSB)
        circ.x(sum_a_reg[n-1])
        circ.cx(sum_a_reg[n-1], cmp_reg[0])
        circ.x(sum_a_reg[n-1])

        # Restore sum_a by adding sum_a_prime back
        self._quantum_add_inplace(circ, sum_a_prime_reg, sum_a_reg, scratch)

    def _copy_winner(self, circ, a_reg, s_reg, cmp_reg, bit_to_flip):
        """
        Copy the winner value to s:
        - If cmp=1: s = a (a wins)
        - If cmp=0: s = a ⊕ 2^{bit_to_flip} (a' wins)

        Implementation:
        1. Copy a to s unconditionally
        2. If cmp=0, flip bit bit_to_flip of s
        """
        # Copy a to s
        for i in range(len(a_reg)):
            circ.cx(a_reg[i], s_reg[i])

        # If cmp=0, flip bit bit_to_flip of s
        # This means: if NOT(cmp), flip s[bit_to_flip]
        circ.x(cmp_reg[0])
        circ.cx(cmp_reg[0], s_reg[bit_to_flip])
        circ.x(cmp_reg[0])

    def run(self, round_num: int = None, shots: int = 1024) -> Tuple[int, dict]:
        """
        Run the quantum algorithm.

        Returns:
            (most_likely_winner, measurement_counts)
        """
        circ, registers = self.build_circuit(round_num)

        # Transpile to decompose multi-controlled gates
        transpiled = transpile(
            circ,
            basis_gates=['cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 'h',
                        's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'swap', 'ccx', 'cz', 'cp']
        )

        print(f"Circuit depth: {transpiled.depth()}, gates: {transpiled.size()}")
        print(f"Qubits: {transpiled.num_qubits}")

        # Use matrix product state simulator for large circuits
        sim = Aer.get_backend('aer_simulator')
        sim.set_options(method='matrix_product_state')

        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()

        # Find most common measurement
        best_count = 0
        best_s = 0

        for bitstring, count in counts.items():
            s_val = int(bitstring.replace(' ', ''), 2)
            if count > best_count:
                best_count = count
                best_s = s_val

        return best_s, counts


def compute_quantum_valuation_sum(data: List[DataPoint], a: int, residual_bits: int) -> int:
    """
    Compute what the quantum circuit should produce for valuation sum.

    This matches the quantum trailing zeros computation, not the classical
    function which uses 1000 for zero residuals.
    """
    total = 0
    for pt in data:
        residual = pt.y - a * pt.x

        if residual == 0:
            # All zeros → valuation = residual_bits
            total += residual_bits
        else:
            # Count trailing zeros in two's complement representation
            # For negative numbers, use two's complement
            if residual < 0:
                residual = (1 << residual_bits) + residual  # Two's complement

            # Count trailing zeros
            count = 0
            temp = residual
            while temp > 0 and (temp & 1) == 0:
                count += 1
                temp >>= 1
            total += count

    return total


def test_full_quantum():
    """Test the fully quantum implementation."""
    print("=" * 60)
    print("Fully Quantum Ladder Algorithm Test")
    print("=" * 60)

    # Simple test: y = 2x, optimal coefficient is 2
    data = [
        DataPoint(1, 2),
        DataPoint(2, 4),
    ]

    print("\nDataset: y = 2x")
    print("Expected optimal coefficient: a* = 2")

    # Compute register sizes for accurate quantum valuation prediction
    max_x = max(pt.x for pt in data)
    max_y = max(pt.y for pt in data)
    max_product = 4 * max_x  # 2^2 * max_x for 2-bit coefficients
    max_residual = max(max_y, max_product) * 2
    residual_bits = max(4, initialise.number_of_bits_required(max_residual) + 1)

    # Show quantum valuations (what the circuit will compute)
    print(f"\nQuantum valuation sums (residual_bits={residual_bits}):")
    for a in range(4):
        val_sum = compute_quantum_valuation_sum(data, a, residual_bits)
        residuals = [pt.y - a * pt.x for pt in data]
        print(f"  a={a}: residuals={residuals}, quantum_valuation_sum={val_sum}")

    # Show expected winners for round 2
    print("\nExpected winners for round 2 (flip bit 1):")
    bit_to_flip = 1
    for a in range(4):
        a_prime = a ^ (1 << bit_to_flip)
        f_a = compute_quantum_valuation_sum(data, a, residual_bits)
        f_a_prime = compute_quantum_valuation_sum(data, a_prime, residual_bits)
        winner = a if f_a >= f_a_prime else a_prime
        print(f"  a={a} vs a'={a_prime}: F(a)={f_a}, F(a')={f_a_prime} → winner={winner}")

    # Build and run quantum circuit
    print("\n--- Running Fully Quantum Circuit ---")
    ladder = FullQuantumLadder(data, coeff_bits=2)

    try:
        winner, counts = ladder.run(round_num=2, shots=1024)

        print(f"\nMeasurement results: {counts}")
        print(f"Most likely winner: s = {winner}")

        # Analyze results
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        print("\nTop results:")
        for bitstring, count in sorted_counts[:4]:
            val = int(bitstring, 2)
            print(f"  s={val}: {count} shots ({100*count/1024:.1f}%)")

        if winner == 2:
            print("\n✓ Correct! Found optimal coefficient a* = 2")
        else:
            print(f"\n✗ Expected a* = 2, got {winner}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


def test_statevector_small():
    """Test with statevector simulation on tiny circuit."""
    print("\n" + "=" * 60)
    print("Statevector Analysis (before measurement)")
    print("=" * 60)

    data = [DataPoint(1, 2), DataPoint(2, 4)]
    ladder = FullQuantumLadder(data, coeff_bits=2)

    circ, registers = ladder.build_circuit(round_num=2)

    # Remove measurement for statevector analysis
    circ_no_measure = circ.remove_final_measurements(inplace=False)

    print(f"\nCircuit stats:")
    print(f"  Qubits: {circ.num_qubits}")
    print(f"  Depth: {circ.depth()}")

    # Try statevector if circuit is small enough
    if circ.num_qubits <= 20:
        try:
            sv = Statevector.from_instruction(circ_no_measure)
            print("\nStatevector computed successfully")

            # Find non-zero amplitudes for s register
            k = ladder.k
            s_probs = {}
            for i in range(len(sv)):
                amp = sv.data[i]
                if abs(amp) > 0.01:
                    # Extract s value from state index
                    # s register is the second k qubits after a register
                    s_val = (i >> k) & ((1 << k) - 1)
                    if s_val not in s_probs:
                        s_probs[s_val] = 0
                    s_probs[s_val] += abs(amp)**2

            print("\nProbability distribution for s:")
            for s, prob in sorted(s_probs.items()):
                print(f"  P(s={s}) = {prob:.4f}")

        except Exception as e:
            print(f"Statevector analysis failed: {e}")
    else:
        print(f"Circuit too large for statevector ({circ.num_qubits} qubits)")


def test_copy_winner_only():
    """Test just the copy_winner function with known cmp values."""
    print("\n" + "=" * 60)
    print("Testing Copy Winner Only")
    print("=" * 60)

    k = 2

    a_reg = QuantumRegister(k, 'a')
    s_reg = QuantumRegister(k, 's')
    cmp_reg = QuantumRegister(1, 'cmp')
    s_classical = ClassicalRegister(k, 's_c')

    # Test 1: cmp = 1 (a wins) → s should equal a
    print("\nTest 1: cmp=1 (a wins), a=2 → s should be 2")
    circ = QuantumCircuit(a_reg, s_reg, cmp_reg, s_classical)
    # Set a = 2 (binary 10)
    circ.x(a_reg[1])
    # Set cmp = 1
    circ.x(cmp_reg[0])

    # Copy winner
    bit_to_flip = 1
    # Copy a to s
    for i in range(k):
        circ.cx(a_reg[i], s_reg[i])
    # If cmp=0, flip bit_to_flip of s
    circ.x(cmp_reg[0])
    circ.cx(cmp_reg[0], s_reg[bit_to_flip])
    circ.x(cmp_reg[0])

    circ.measure(s_reg, s_classical)

    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts}")

    # Test 2: cmp = 0 (a' wins) → s should equal a XOR 2
    print("\nTest 2: cmp=0 (a' wins), a=0 → s should be 0 XOR 2 = 2")
    circ = QuantumCircuit(a_reg, s_reg, cmp_reg, s_classical)
    # a = 0 (all zeros by default)
    # cmp = 0 (default)

    # Copy winner
    for i in range(k):
        circ.cx(a_reg[i], s_reg[i])
    circ.x(cmp_reg[0])
    circ.cx(cmp_reg[0], s_reg[bit_to_flip])
    circ.x(cmp_reg[0])

    circ.measure(s_reg, s_classical)
    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts}")


def test_comparison_only():
    """Test just the comparison function."""
    print("\n" + "=" * 60)
    print("Testing Comparison Only")
    print("=" * 60)

    sum_bits = 4

    sum_a_reg = QuantumRegister(sum_bits, 'sum_a')
    sum_a_prime_reg = QuantumRegister(sum_bits, 'sum_ap')
    cmp_reg = QuantumRegister(1, 'cmp')
    scratch = QuantumRegister(sum_bits + 2, 'scratch')
    cmp_classical = ClassicalRegister(1, 'cmp_c')

    # Create dummy ladder for methods
    data = [DataPoint(1, 2)]
    ladder = FullQuantumLadder(data, coeff_bits=2)

    # Test: sum_a = 5, sum_a_prime = 3 → sum_a > sum_a_prime → cmp should be 1
    print("\nTest: sum_a=5, sum_a_prime=3 → cmp should be 1")
    circ = QuantumCircuit(sum_a_reg, sum_a_prime_reg, cmp_reg, scratch, cmp_classical)

    # Set sum_a = 5 (binary 0101)
    circ.x(sum_a_reg[0])
    circ.x(sum_a_reg[2])

    # Set sum_a_prime = 3 (binary 0011)
    circ.x(sum_a_prime_reg[0])
    circ.x(sum_a_prime_reg[1])

    ladder._compare_sums(circ, sum_a_reg, sum_a_prime_reg, cmp_reg, scratch)
    circ.measure(cmp_reg, cmp_classical)

    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts} (expected: cmp=1)")

    # Test: sum_a = 2, sum_a_prime = 7 → sum_a < sum_a_prime → cmp should be 0
    print("\nTest: sum_a=2, sum_a_prime=7 → cmp should be 0")
    circ = QuantumCircuit(sum_a_reg, sum_a_prime_reg, cmp_reg, scratch, cmp_classical)

    # Set sum_a = 2 (binary 0010)
    circ.x(sum_a_reg[1])

    # Set sum_a_prime = 7 (binary 0111)
    circ.x(sum_a_prime_reg[0])
    circ.x(sum_a_prime_reg[1])
    circ.x(sum_a_prime_reg[2])

    ladder._compare_sums(circ, sum_a_reg, sum_a_prime_reg, cmp_reg, scratch)
    circ.measure(cmp_reg, cmp_classical)

    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts} (expected: cmp=0)")


def test_comparison_logic():
    """Test the comparison and copy logic together."""
    print("\n" + "=" * 60)
    print("Testing Comparison and Copy Logic Together")
    print("=" * 60)

    k = 2
    sum_bits = 4

    a_reg = QuantumRegister(k, 'a')
    s_reg = QuantumRegister(k, 's')
    sum_a_reg = QuantumRegister(sum_bits, 'sum_a')
    sum_a_prime_reg = QuantumRegister(sum_bits, 'sum_ap')
    cmp_reg = QuantumRegister(1, 'cmp')
    scratch = QuantumRegister(sum_bits + 2, 'scratch')
    s_classical = ClassicalRegister(k, 's_c')

    data = [DataPoint(1, 2)]
    ladder = FullQuantumLadder(data, coeff_bits=k)

    # Test with a single case: a=0, sum_a=2, sum_a_prime=5
    # sum_a < sum_a_prime → a' wins → s = 0 XOR 2 = 2
    print("\nTest: a=0, sum_a=2, sum_a_prime=5 → cmp=0 → s=2")

    circ = QuantumCircuit(a_reg, s_reg, sum_a_reg, sum_a_prime_reg, cmp_reg, scratch, s_classical)

    # a = 0 (default)
    # sum_a = 2
    circ.x(sum_a_reg[1])
    # sum_a_prime = 5
    circ.x(sum_a_prime_reg[0])
    circ.x(sum_a_prime_reg[2])

    # Compare
    ladder._compare_sums(circ, sum_a_reg, sum_a_prime_reg, cmp_reg, scratch)

    # Check cmp value with statevector
    sv = Statevector.from_instruction(circ)
    # Total qubits: 2 + 2 + 4 + 4 + 1 + 6 = 19
    # cmp is at bit position 2 + 2 + 4 + 4 = 12
    cmp_bit_pos = k + k + sum_bits + sum_bits
    print(f"  cmp bit position: {cmp_bit_pos}")

    for i in range(len(sv)):
        if abs(sv.data[i]) > 0.01:
            cmp_val = (i >> cmp_bit_pos) & 1
            print(f"  State {i}: cmp={cmp_val}, amp={sv.data[i]:.3f}")

    # Copy winner
    bit_to_flip = 1
    ladder._copy_winner(circ, a_reg, s_reg, cmp_reg, bit_to_flip)

    circ.measure(s_reg, s_classical)

    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Measurement result: {counts} (expected s=2 → '10')")


def test_subtraction():
    """Test the quantum subtraction directly."""
    print("\n" + "=" * 60)
    print("Testing Quantum Subtraction")
    print("=" * 60)

    n_bits = 4

    a_reg = QuantumRegister(n_bits, 'a')
    b_reg = QuantumRegister(n_bits, 'b')
    scratch = QuantumRegister(n_bits + 2, 'scratch')
    b_classical = ClassicalRegister(n_bits, 'b_c')

    data = [DataPoint(1, 2)]
    ladder = FullQuantumLadder(data, coeff_bits=2)

    # Test: b = 5, a = 3 → b - a = 2
    print("\nTest: b=5, a=3 → b-a should be 2")
    circ = QuantumCircuit(a_reg, b_reg, scratch, b_classical)

    # Set a = 3 (0011)
    circ.x(a_reg[0])
    circ.x(a_reg[1])

    # Set b = 5 (0101)
    circ.x(b_reg[0])
    circ.x(b_reg[2])

    # Subtract: b = b - a
    ladder._quantum_subtract_inplace(circ, a_reg, b_reg, scratch)

    circ.measure(b_reg, b_classical)

    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts} (expected b=2 → '0010')")

    # Test: b = 3, a = 5 → b - a = -2 = 14 (in 4-bit two's complement)
    print("\nTest: b=3, a=5 → b-a should be -2 = 14 (1110)")
    circ = QuantumCircuit(a_reg, b_reg, scratch, b_classical)

    # Set a = 5 (0101)
    circ.x(a_reg[0])
    circ.x(a_reg[2])

    # Set b = 3 (0011)
    circ.x(b_reg[0])
    circ.x(b_reg[1])

    ladder._quantum_subtract_inplace(circ, a_reg, b_reg, scratch)

    circ.measure(b_reg, b_classical)

    result = sim.run(circ, shots=100).result()
    counts = result.get_counts()
    print(f"  Result: {counts} (expected b=-2 → '1110')")

    # Check MSB directly
    print("\nChecking MSB after subtraction b=3-5:")
    circ = QuantumCircuit(a_reg, b_reg, scratch)

    circ.x(a_reg[0])
    circ.x(a_reg[2])
    circ.x(b_reg[0])
    circ.x(b_reg[1])

    ladder._quantum_subtract_inplace(circ, a_reg, b_reg, scratch)

    sv = Statevector.from_instruction(circ)
    print(f"  Statevector shows only one state:")
    for i in range(len(sv)):
        if abs(sv.data[i]) > 0.01:
            b_val = (i >> n_bits) & 15  # b is at bits 4-7
            print(f"    b = {b_val} = {bin(b_val)}, MSB = {(b_val >> 3) & 1}")


def test_residual_computation():
    """Test the quantum residual sum computation for a single coefficient."""
    print("\n" + "=" * 60)
    print("Testing Residual Sum Computation")
    print("=" * 60)

    # Simple test: y = 2x with just one data point
    data = [DataPoint(1, 2)]
    k = 2

    ladder = FullQuantumLadder(data, coeff_bits=k)

    # We'll test F(0), F(1), F(2), F(3)
    for a_val in range(4):
        # Create circuit with a fixed to a_val
        a_reg = QuantumRegister(ladder.k, 'a')
        sum_reg = QuantumRegister(ladder.sum_bits, 'sum')
        product_reg = QuantumRegister(ladder.product_bits, 'prod')
        residual_reg = QuantumRegister(ladder.residual_bits, 'resid')
        valuation_reg = QuantumRegister(ladder.valuation_bits, 'val')
        arith_scratch = QuantumRegister(max(ladder.product_bits, ladder.residual_bits, ladder.sum_bits) + 2, 'arith')
        tz_scratch = QuantumRegister(1 + 2 + ladder.valuation_bits + 1 + ladder.residual_bits, 'tz')
        sum_classical = ClassicalRegister(ladder.sum_bits, 'sum_c')

        circ = QuantumCircuit(a_reg, sum_reg, product_reg, residual_reg, valuation_reg,
                              arith_scratch, tz_scratch, sum_classical)

        # Set a to a_val
        for i in range(k):
            if (a_val >> i) & 1:
                circ.x(a_reg[i])

        # Compute residual sum
        ladder._compute_residual_sum(circ, a_reg, sum_reg,
                                      product_reg, residual_reg, valuation_reg,
                                      arith_scratch, tz_scratch)

        circ.measure(sum_reg, sum_classical)

        # Transpile for MPS
        transpiled = transpile(circ, basis_gates=['cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 'h',
                                                  's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'swap', 'ccx', 'cz', 'cp'])
        sim = Aer.get_backend('aer_simulator')
        sim.set_options(method='matrix_product_state')
        result = sim.run(transpiled, shots=100).result()
        counts = result.get_counts()

        expected = compute_quantum_valuation_sum(data, a_val, ladder.residual_bits)
        measured = int(list(counts.keys())[0], 2)
        status = "✓" if measured == expected else "✗"
        print(f"  a={a_val}: measured F(a)={measured}, expected={expected} {status}")


if __name__ == "__main__":
    test_residual_computation()  # Test F(a) computation first
    test_subtraction()
    test_copy_winner_only()
    test_comparison_only()
    test_comparison_logic()
    # Basic tests pass, now run full quantum:
    test_full_quantum()
