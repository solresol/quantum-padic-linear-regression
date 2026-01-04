#!/usr/bin/env python3
"""
Minimal 2-bit Quantum Ladder Debug Version

This is a simplified implementation for debugging the quantum ladder algorithm.
Uses only 2-bit coefficients (a ∈ {0,1,2,3}) to make the state evolution tractable.

Key change from v2: Instead of setting s=0 or s=1 based on "which is better",
we set s to the actual winning bit value (the value of bit t-1 that gives
the lower residual sum).

Data format: y = a*x (no intercept)
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import math
from typing import List, Tuple

from padic_core import DataPoint, classical_2adic_valuation, classical_valuation_sum


def compute_valuation_sum_classical(data: List[DataPoint], a: int) -> int:
    """
    Classically compute sum of 2-adic valuations for coefficient a.
    Higher is better (means smaller 2-adic distances).
    """
    total = 0
    for pt in data:
        residual = pt.y - a * pt.x
        if residual == 0:
            total += 100  # Large value for zero residual
        else:
            total += classical_2adic_valuation(abs(residual))
    return total


def build_simple_oracle(data: List[DataPoint], t: int, k: int = 2) -> QuantumCircuit:
    """
    Build a simple oracle that uses classical precomputation.

    For debugging, we compute valuations classically and encode them
    as phase kickback. This isolates whether the algorithm structure is correct.

    In round t, for each state |a⟩ we need to:
    1. Compare F(a) vs F(a ⊕ 2^{t-1})
    2. Set s = winning bit value at position t-1
    3. Use s to apply phase kickback

    Args:
        data: Dataset (y = a*x)
        t: Round number (1 = find bit 0, 2 = find bit 1)
        k: Number of coefficient bits (default 2)

    Returns:
        Quantum circuit for this round
    """
    a_reg = QuantumRegister(k, 'a')
    s_reg = QuantumRegister(1, 's')  # Stores winning bit value
    a_classical = ClassicalRegister(k, 'a_c')

    circ = QuantumCircuit(a_reg, s_reg, a_classical)

    # Step 1: Superposition over all coefficient values
    for qubit in a_reg:
        circ.h(qubit)

    circ.barrier(label='superposition')

    # Step 2: For each basis state |a⟩, compute which bit value wins
    # We use controlled operations based on the computational basis state
    bit_pos = t - 1  # Bit position we're determining in this round

    # For each possible value of a, we need to:
    # - Compute a' = a XOR 2^{bit_pos}
    # - Compare F(a) vs F(a')
    # - Set s to the winning bit value at position bit_pos

    # Precompute: for each a in {0, 1, 2, 3}, what is the winning bit at bit_pos?
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)  # a' = a XOR 2^{bit_pos}

        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        # Higher valuation sum is better
        if f_a >= f_a_flip:
            winning_bit = (a >> bit_pos) & 1  # a's bit is the winner
        else:
            winning_bit = (a_flip >> bit_pos) & 1  # a_flip's bit is the winner

        # Set s = winning_bit when a_reg == a
        # This requires multi-controlled gates
        if winning_bit == 1:
            # Apply X to s when a_reg == a
            controls = []
            for i in range(k):
                if (a >> i) & 1:
                    controls.append(a_reg[i])
                else:
                    circ.x(a_reg[i])
                    controls.append(a_reg[i])

            # Multi-controlled X on s
            if len(controls) == 1:
                circ.cx(controls[0], s_reg[0])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], s_reg[0])
            else:
                circ.mcx(controls, s_reg[0])

            # Undo X flips
            for i in range(k):
                if not ((a >> i) & 1):
                    circ.x(a_reg[i])

    circ.barrier(label='oracle')

    # Step 3: Apply phase kickback based on s
    # States where s=1 get a phase flip
    circ.z(s_reg[0])

    circ.barrier(label='phase')

    # Step 4: Uncompute s (run oracle again to reset s to 0)
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)

        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        if f_a >= f_a_flip:
            winning_bit = (a >> bit_pos) & 1
        else:
            winning_bit = (a_flip >> bit_pos) & 1

        if winning_bit == 1:
            controls = []
            for i in range(k):
                if (a >> i) & 1:
                    controls.append(a_reg[i])
                else:
                    circ.x(a_reg[i])
                    controls.append(a_reg[i])

            if len(controls) == 1:
                circ.cx(controls[0], s_reg[0])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], s_reg[0])
            else:
                circ.mcx(controls, s_reg[0])

            for i in range(k):
                if not ((a >> i) & 1):
                    circ.x(a_reg[i])

    circ.barrier(label='uncompute')

    # Step 5: Apply QFT
    n = k
    for i in range(n):
        circ.h(a_reg[i])
        for j in range(i + 1, n):
            angle = math.pi / (1 << (j - i))
            circ.cp(angle, a_reg[j], a_reg[i])

    # Swap for standard QFT ordering
    for i in range(n // 2):
        circ.swap(a_reg[i], a_reg[n - 1 - i])

    circ.barrier(label='QFT')

    # Step 6: Measure
    circ.measure(a_reg, a_classical)

    return circ


def analyze_oracle_behavior(data: List[DataPoint], t: int, k: int = 2):
    """
    Analyze what the oracle does for each basis state.
    """
    bit_pos = t - 1
    print(f"\n=== Oracle Analysis for Round {t} (bit {bit_pos}) ===")
    print(f"Data: {[(pt.x, pt.y) for pt in data]}")

    # For each possible a value, compute valuations and determine winner
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)

        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        a_bit = (a >> bit_pos) & 1
        a_flip_bit = (a_flip >> bit_pos) & 1

        if f_a >= f_a_flip:
            winning_bit = a_bit
            winner = "a"
        else:
            winning_bit = a_flip_bit
            winner = "a'"

        print(f"  a={a} (bit{bit_pos}={a_bit}): F(a)={f_a}, "
              f"a'={a_flip} (bit{bit_pos}={a_flip_bit}): F(a')={f_a_flip}, "
              f"winner={winner}, s={winning_bit}")

    # Show which phase flips happen
    print("\nPhase analysis:")
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)
        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        if f_a >= f_a_flip:
            winning_bit = (a >> bit_pos) & 1
        else:
            winning_bit = (a_flip >> bit_pos) & 1

        phase = "-" if winning_bit == 1 else "+"
        print(f"  |{a}⟩ gets phase {phase}")


def run_statevector_analysis(data: List[DataPoint], t: int, k: int = 2):
    """
    Run the circuit with statevector simulation to see exact amplitudes.
    """
    print(f"\n=== Statevector Analysis for Round {t} ===")

    # Build circuit without measurement for statevector
    a_reg = QuantumRegister(k, 'a')
    s_reg = QuantumRegister(1, 's')

    circ = QuantumCircuit(a_reg, s_reg)

    # Step 1: Superposition
    for qubit in a_reg:
        circ.h(qubit)

    print("After superposition:")
    sv = Statevector.from_instruction(circ)
    print_amplitudes(sv, k)

    # Step 2: Oracle
    bit_pos = t - 1
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)

        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        if f_a >= f_a_flip:
            winning_bit = (a >> bit_pos) & 1
        else:
            winning_bit = (a_flip >> bit_pos) & 1

        if winning_bit == 1:
            controls = []
            for i in range(k):
                if (a >> i) & 1:
                    controls.append(a_reg[i])
                else:
                    circ.x(a_reg[i])
                    controls.append(a_reg[i])

            if len(controls) == 1:
                circ.cx(controls[0], s_reg[0])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], s_reg[0])
            else:
                circ.mcx(controls, s_reg[0])

            for i in range(k):
                if not ((a >> i) & 1):
                    circ.x(a_reg[i])

    print("\nAfter oracle (setting s):")
    sv = Statevector.from_instruction(circ)
    print_amplitudes(sv, k, show_s=True)

    # Step 3: Phase kickback
    circ.z(s_reg[0])

    print("\nAfter phase kickback (Z on s):")
    sv = Statevector.from_instruction(circ)
    print_amplitudes(sv, k, show_s=True)

    # Step 4: Uncompute s
    for a in range(1 << k):
        a_flip = a ^ (1 << bit_pos)

        f_a = compute_valuation_sum_classical(data, a)
        f_a_flip = compute_valuation_sum_classical(data, a_flip)

        if f_a >= f_a_flip:
            winning_bit = (a >> bit_pos) & 1
        else:
            winning_bit = (a_flip >> bit_pos) & 1

        if winning_bit == 1:
            controls = []
            for i in range(k):
                if (a >> i) & 1:
                    controls.append(a_reg[i])
                else:
                    circ.x(a_reg[i])
                    controls.append(a_reg[i])

            if len(controls) == 1:
                circ.cx(controls[0], s_reg[0])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], s_reg[0])
            else:
                circ.mcx(controls, s_reg[0])

            for i in range(k):
                if not ((a >> i) & 1):
                    circ.x(a_reg[i])

    print("\nAfter uncompute s:")
    sv = Statevector.from_instruction(circ)
    print_amplitudes(sv, k, show_s=True)

    # Step 5: QFT
    n = k
    for i in range(n):
        circ.h(a_reg[i])
        for j in range(i + 1, n):
            angle = math.pi / (1 << (j - i))
            circ.cp(angle, a_reg[j], a_reg[i])

    for i in range(n // 2):
        circ.swap(a_reg[i], a_reg[n - 1 - i])

    print("\nAfter QFT:")
    sv = Statevector.from_instruction(circ)
    print_amplitudes(sv, k, show_s=True)

    # Compute measurement probabilities
    print("\nMeasurement probabilities (ignoring s qubit):")
    probs = {}
    for i in range(1 << k):
        # Sum probabilities over s=0 and s=1
        prob = abs(sv[i])**2 + abs(sv[i + (1 << k)])**2
        probs[i] = prob
        print(f"  |a={i}⟩: {prob:.4f}")

    return probs


def print_amplitudes(sv, k, show_s=False):
    """Print non-zero amplitudes."""
    n_states = 1 << k
    for i in range(len(sv)):
        amp = sv[i]
        if abs(amp) > 1e-10:
            if show_s:
                s_val = i >> k
                a_val = i & ((1 << k) - 1)
                print(f"  |a={a_val}, s={s_val}⟩: {amp:.4f}")
            else:
                print(f"  |{i}⟩: {amp:.4f}")


def explore_phase_patterns():
    """
    Explore what phase patterns would give correct interference.

    Key insight: After QFT, we want to measure the optimal BIT VALUE, not
    the optimal COEFFICIENT. So we need a phase pattern that encodes
    "which bit value is optimal at position t-1" in a way that QFT extracts it.
    """
    print("\n" + "=" * 60)
    print("Exploring Phase Pattern / QFT Relationship")
    print("=" * 60)

    from qiskit.quantum_info import Statevector
    import numpy as np

    k = 2

    def get_amplitudes(sv, n=4):
        """Extract amplitudes from statevector."""
        data = sv.data
        return [complex(round(data[i].real, 3), round(data[i].imag, 3)) for i in range(min(n, len(data)))]

    # What we want: after QFT, measure bit t-1 of the result = optimal bit at position t-1

    # For round 1 (t=1), optimal bit 0 = 0 (since a*=2=10 binary)
    # For round 2 (t=2), optimal bit 1 = 1 (since a*=2=10 binary)

    print("\nTest 1: Mark states where bit 0 = 0 (optimal)")
    print("  States: |0⟩, |2⟩")
    circ = QuantumCircuit(2)
    for i in range(2):
        circ.h(i)
    # Apply phase to states where bit 0 = 0: a ∈ {0, 2}
    # Control on NOT bit 0
    circ.x(0)
    circ.cz(0, 1)  # This flips phase when both controls are 1
    circ.z(0)  # Also flip when just bit 0 control is 1
    circ.x(0)
    sv = Statevector.from_instruction(circ)
    print(f"  After marking: {get_amplitudes(sv)}")

    # QFT
    circ.h(0)
    circ.cp(math.pi/2, 1, 0)
    circ.h(1)
    circ.swap(0, 1)
    sv = Statevector.from_instruction(circ)
    print(f"  After QFT: {get_amplitudes(sv)}")
    print(f"  Probabilities: {[round(abs(sv.data[i])**2, 4) for i in range(4)]}")

    print("\nTest 2: Mark states where bit 1 = 1 (optimal for a*=2)")
    print("  States: |2⟩, |3⟩")
    circ = QuantumCircuit(2)
    for i in range(2):
        circ.h(i)
    # Apply phase to states where bit 1 = 1: a ∈ {2, 3}
    # Z(1) applies -1 to |2⟩ and |3⟩
    circ.z(1)
    sv = Statevector.from_instruction(circ)
    print(f"  After marking: {get_amplitudes(sv)}")

    # QFT
    circ.h(0)
    circ.cp(math.pi/2, 1, 0)
    circ.h(1)
    circ.swap(0, 1)
    sv = Statevector.from_instruction(circ)
    print(f"  After QFT: {get_amplitudes(sv)}")
    print(f"  Probabilities: {[round(abs(sv.data[i])**2, 4) for i in range(4)]}")

    # The key insight: what if we apply phase based on the WINNING BIT VALUE
    # relative to the qubit we're trying to determine?

    print("\n" + "=" * 60)
    print("Alternative: Phase on bit t-1 conditioned on oracle outcome")
    print("=" * 60)

    print("\nRound 2 reimagined: Apply phase to bit 1 when it has wrong value")
    circ = QuantumCircuit(2)
    for i in range(2):
        circ.h(i)

    # For each state, we know the optimal bit 1 value:
    #   a=0: optimal bit 1 = 1 (winner is a'=2)
    #   a=1: optimal bit 1 = 0 (winner is a=1, tie)
    #   a=2: optimal bit 1 = 1 (winner is a=2)
    #   a=3: optimal bit 1 = 1 (winner is a=3, tie)

    # So states where current bit 1 ≠ optimal bit 1:
    #   a=0: bit 1 = 0, optimal = 1 → WRONG
    #   a=1: bit 1 = 0, optimal = 0 → RIGHT
    #   a=2: bit 1 = 1, optimal = 1 → RIGHT
    #   a=3: bit 1 = 1, optimal = 1 → RIGHT

    # Apply phase to |0⟩ only (the only "wrong" state)
    circ.x(0)
    circ.x(1)
    circ.cz(0, 1)
    circ.x(0)
    circ.x(1)
    sv = Statevector.from_instruction(circ)
    print(f"  After marking WRONG states: {get_amplitudes(sv)}")

    # Apply H only to qubit 1 (the bit we're determining)
    circ.h(1)
    sv = Statevector.from_instruction(circ)
    print(f"  After H on bit 1: {get_amplitudes(sv)}")

    # Probabilities for bit 1
    data = sv.data
    p_bit1_0 = abs(data[0])**2 + abs(data[1])**2  # |00⟩ + |01⟩
    p_bit1_1 = abs(data[2])**2 + abs(data[3])**2  # |10⟩ + |11⟩
    print(f"  P(bit 1 = 0) = {p_bit1_0:.4f}")
    print(f"  P(bit 1 = 1) = {p_bit1_1:.4f}")

    print("\n" + "=" * 60)
    print("Key Insight: The problem with small search spaces")
    print("=" * 60)

    # The issue is that in small spaces, pairs not containing the optimum
    # can give inconsistent information.

    # For our 4-state example:
    # - Pair (0,2) includes optimum 2 → gives correct bit 1 info
    # - Pair (1,3) doesn't include optimum → gives inconsistent info

    print("\nPair analysis for bit 1:")
    print("  Pair (a=0, a'=2): F(0)=3, F(2)=200 → bit 1 should be 1 (correct)")
    print("  Pair (a=1, a'=3): F(1)=1, F(3)=1 → TIE, bit 1 undetermined")
    print("")
    print("With large search spaces, pairs containing the optimum dominate.")
    print("With 4 states, distractor pairs have equal weight → interference fails.")

    print("\n" + "=" * 60)
    print("Alternative: Continuous phase encoding (phase estimation style)")
    print("=" * 60)

    # Instead of binary phase flips, encode phase proportional to valuation sum
    # Better states (higher F) get larger phase rotation
    # After inverse QFT, optimal should concentrate

    data = [DataPoint(1, 2), DataPoint(2, 4)]

    # Get valuations
    vals = [compute_valuation_sum_classical(data, a) for a in range(4)]
    max_val = max(vals)
    print(f"\nValuations: {vals}")
    print(f"Normalized: {[v/max_val for v in vals]}")

    circ = QuantumCircuit(2)
    for i in range(2):
        circ.h(i)

    # Apply phase proportional to F(a)/F_max
    for a in range(4):
        phase = math.pi * vals[a] / max_val  # Phase from 0 to π

        # Apply controlled phase for state |a⟩
        controls_needed = []
        for i in range(2):
            if (a >> i) & 1:
                controls_needed.append((i, False))  # Need qubit i = 1
            else:
                controls_needed.append((i, True))   # Need qubit i = 0

        # Build the controlled phase
        for i, flip_first in controls_needed:
            if flip_first:
                circ.x(i)

        # For 2 qubits, use conditional phase
        if len(controls_needed) == 2:
            # CP between qubits, with phase if both match
            circ.cp(phase, 0, 1)

        for i, flip_first in controls_needed:
            if flip_first:
                circ.x(i)

    sv = Statevector.from_instruction(circ)
    print(f"\nAfter phase encoding:")
    for a in range(4):
        amp = sv.data[a]
        print(f"  |{a}⟩: {amp:.4f} (phase = {math.atan2(amp.imag, amp.real):.3f})")

    # Inverse QFT would be: swap, then inverse phases and H
    # For simplicity, let's try regular QFT and see what happens
    circ.h(0)
    circ.cp(math.pi/2, 1, 0)
    circ.h(1)
    circ.swap(0, 1)

    sv = Statevector.from_instruction(circ)
    print(f"\nAfter QFT:")
    probs = {}
    for a in range(4):
        probs[a] = abs(sv.data[a])**2
        print(f"  |{a}⟩: prob = {probs[a]:.4f}")

    print(f"\nMost likely outcome: {max(probs, key=probs.get)}")

    print("\n" + "=" * 60)
    print("User's suggestion: Store winning VALUE, not just 0/1")
    print("=" * 60)

    # Instead of s = 0 or 1 (winning bit), store s = winning VALUE
    # This gives constructive interference when multiple a states
    # have the same winner

    print("\nFor round 2, storing winning coefficient value in s register:")
    print("  |a=0⟩ → winner is a'=2, so s=10 (binary for 2)")
    print("  |a=1⟩ → winner is a=1, so s=01 (binary for 1)")
    print("  |a=2⟩ → winner is a=2, so s=10 (binary for 2)")
    print("  |a=3⟩ → winner is a=3, so s=11 (binary for 3)")

    # Build circuit with 2-qubit s register
    a_reg = QuantumRegister(2, 'a')
    s_reg = QuantumRegister(2, 's')  # Now 2 qubits to store value 0-3
    s_classical = ClassicalRegister(2, 's_c')

    circ = QuantumCircuit(a_reg, s_reg, s_classical)

    # Superposition on a
    circ.h(a_reg[0])
    circ.h(a_reg[1])

    # For each a, copy the winning value to s
    # This requires computing the comparison and conditionally copying

    # a=0: if a==0, set s=2 (since a'=2 wins)
    # Control: a[0]=0, a[1]=0
    circ.x(a_reg[0])
    circ.x(a_reg[1])
    # Set s = 2 = 10 binary
    circ.ccx(a_reg[0], a_reg[1], s_reg[1])  # Set s[1]=1
    circ.x(a_reg[0])
    circ.x(a_reg[1])

    # a=1: if a==1, set s=1 (since a=1 wins by tie)
    # Control: a[0]=1, a[1]=0
    circ.x(a_reg[1])
    circ.ccx(a_reg[0], a_reg[1], s_reg[0])  # Set s[0]=1
    circ.x(a_reg[1])

    # a=2: if a==2, set s=2 (since a=2 wins)
    # Control: a[0]=0, a[1]=1
    circ.x(a_reg[0])
    circ.ccx(a_reg[0], a_reg[1], s_reg[1])  # Set s[1]=1
    circ.x(a_reg[0])

    # a=3: if a==3, set s=3 (since a=3 wins by tie)
    # Control: a[0]=1, a[1]=1
    circ.ccx(a_reg[0], a_reg[1], s_reg[0])  # Set s[0]=1
    circ.ccx(a_reg[0], a_reg[1], s_reg[1])  # Set s[1]=1

    sv = Statevector.from_instruction(circ)
    print("\nState after encoding winners:")
    for i in range(16):
        if abs(sv.data[i]) > 0.01:
            s_val = i >> 2
            a_val = i & 3
            print(f"  |a={a_val}, s={s_val}⟩: {sv.data[i]:.4f}")

    # Measure s
    circ.measure(s_reg, s_classical)

    sim = Aer.get_backend('aer_simulator')
    result = sim.run(circ, shots=1024).result()
    counts = result.get_counts()
    print(f"\nMeasurement of s register: {counts}")
    print("Expected: s=2 should have ~50% (from a=0 and a=2 both giving winner 2)")


def test_minimal():
    """Test with minimal 2-bit case."""
    print("=" * 60)
    print("Minimal 2-bit Quantum Ladder Debug")
    print("=" * 60)

    # Data: y = 2x (optimal coefficient is 2)
    data = [
        DataPoint(1, 2),
        DataPoint(2, 4),
    ]

    k = 2  # 2-bit coefficients: a ∈ {0, 1, 2, 3}

    # Show classical valuations
    print("\nClassical valuations for each coefficient:")
    for a in range(1 << k):
        val_sum = compute_valuation_sum_classical(data, a)
        residuals = [pt.y - a * pt.x for pt in data]
        print(f"  a={a}: residuals={residuals}, valuation_sum={val_sum}")

    # Analyze oracle behavior for each round
    for t in [1, 2]:
        analyze_oracle_behavior(data, t, k)
        run_statevector_analysis(data, t, k)

    # Explore phase patterns
    explore_phase_patterns()

    # Run actual circuit with shots
    print("\n" + "=" * 60)
    print("Running circuits with shots")
    print("=" * 60)

    for t in [1, 2]:
        print(f"\n--- Round {t} ---")
        circ = build_simple_oracle(data, t, k)

        sim = Aer.get_backend('aer_simulator')
        result = sim.run(circ, shots=1024).result()
        counts = result.get_counts()

        print(f"Measurement counts: {counts}")

        # Find most common
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        best = sorted_counts[0]
        print(f"Most likely: {best}")


if __name__ == "__main__":
    test_minimal()
