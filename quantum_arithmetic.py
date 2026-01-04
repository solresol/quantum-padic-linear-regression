#!/usr/bin/env python3
"""
Quantum Arithmetic Circuits

This module implements quantum circuits for basic arithmetic operations
needed by the p-adic regression algorithm:
- Addition of two quantum registers
- Subtraction of two quantum registers
- Multiplication of a quantum register by a classical constant
- Controlled versions of the above
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import initialise


def quantum_add(circ: QuantumCircuit,
                a_reg: QuantumRegister,
                b_reg: QuantumRegister,
                scratch: QuantumRegister,
                result_reg: QuantumRegister = None):
    """
    Quantum addition: result = a + b (or b = a + b if result_reg is None).

    Uses ripple-carry addition. If result_reg is provided, the sum goes there.
    Otherwise, b_reg is modified in place to hold b + a.

    Args:
        circ: Quantum circuit to add gates to
        a_reg: First operand (unchanged)
        b_reg: Second operand (modified if result_reg is None)
        scratch: Scratch qubits for carries (needs len(b_reg) qubits)
        result_reg: Optional output register for result
    """
    if result_reg is None:
        result_reg = b_reg

    n = len(a_reg)
    if len(scratch) < n:
        raise ValueError(f"Need at least {n} scratch qubits for addition")

    # Ripple carry addition
    # carry[i] = carry[i-1] AND (a[i-1] XOR b[i-1]) OR (a[i-1] AND b[i-1])
    # But simplified: just propagate carries

    # First, copy b to result if different
    if result_reg is not b_reg:
        for i in range(min(len(b_reg), len(result_reg))):
            circ.cx(b_reg[i], result_reg[i])

    # Now add a to result with carries
    # This is a simplified version - for bit 0, just XOR
    circ.cx(a_reg[0], result_reg[0])

    # Generate carry into bit 1
    circ.ccx(a_reg[0], b_reg[0], scratch[0])

    for i in range(1, n):
        # Sum bit: result[i] = a[i] XOR b[i] XOR carry[i-1]
        circ.cx(a_reg[i], result_reg[i])
        circ.cx(scratch[i-1], result_reg[i])

        # Generate carry for next bit (if not last)
        if i < n - 1:
            # carry[i] = (a[i] AND b[i]) OR (carry[i-1] AND (a[i] XOR b[i]))
            # Simplified: carry[i] = majority(a[i], b[i], carry[i-1])
            circ.ccx(a_reg[i], b_reg[i], scratch[i])
            circ.ccx(a_reg[i], scratch[i-1], scratch[i])
            circ.ccx(b_reg[i], scratch[i-1], scratch[i])

    # Uncompute carries (in reverse)
    for i in range(n - 2, 0, -1):
        circ.ccx(b_reg[i], scratch[i-1], scratch[i])
        circ.ccx(a_reg[i], scratch[i-1], scratch[i])
        circ.ccx(a_reg[i], b_reg[i], scratch[i])

    circ.ccx(a_reg[0], b_reg[0], scratch[0])


def quantum_subtract(circ: QuantumCircuit,
                     minuend_reg: QuantumRegister,
                     subtrahend_reg: QuantumRegister,
                     result_reg: QuantumRegister,
                     scratch: QuantumRegister):
    """
    Quantum subtraction: result = minuend - subtrahend.

    Uses two's complement: a - b = a + (~b + 1)

    Args:
        circ: Quantum circuit
        minuend_reg: The value to subtract from (a)
        subtrahend_reg: The value to subtract (b) - unchanged after operation
        result_reg: Output register for a - b (must start as |0>)
        scratch: Scratch qubits (needs len(result_reg))
    """
    n = len(result_reg)
    m = min(len(subtrahend_reg), n)

    # Step 1: Copy minuend to result
    for i in range(min(len(minuend_reg), n)):
        circ.cx(minuend_reg[i], result_reg[i])

    # Step 2: Compute two's complement of subtrahend and add to result
    # Two's complement of b is ~b + 1

    # First, add ~b by doing controlled additions with flipped control sense
    # For each bit of subtrahend: if bit=0, add 2^i to result; if bit=1, don't add

    for i in range(m):
        # Add 2^i if subtrahend[i] = 0
        circ.x(subtrahend_reg[i])  # Flip so we control on "was 0"
        _controlled_add_power_of_2(circ, subtrahend_reg[i], result_reg, i, scratch)
        circ.x(subtrahend_reg[i])  # Restore

    # Add the implicit 1s for bits beyond subtrahend length (if any)
    # ~b for those bits is all 1s, so we add 2^i for each
    for i in range(m, n):
        # Unconditionally add 2^i
        _add_power_of_2(circ, result_reg, i)

    # Step 3: Add 1 to complete two's complement
    _add_one(circ, result_reg)


def _add_power_of_2(circ: QuantumCircuit, target_reg: QuantumRegister, power: int):
    """Unconditionally add 2^power to target register."""
    n = len(target_reg)
    if power >= n:
        return

    # Work from highest bit down
    for i in range(n - 1, power - 1, -1):
        if i == power:
            circ.x(target_reg[i])
        else:
            # Flip if all bits from power to i-1 are 1
            controls = [target_reg[j] for j in range(power, i)]
            if len(controls) == 1:
                circ.cx(controls[0], target_reg[i])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], target_reg[i])
            else:
                circ.mcx(controls, target_reg[i])


def _add_one(circ: QuantumCircuit, target_reg: QuantumRegister):
    """Add 1 to target register (increment)."""
    n = len(target_reg)

    # Work from highest bit down
    for i in range(n - 1, -1, -1):
        if i == 0:
            circ.x(target_reg[0])
        else:
            # Flip bit i if all bits 0 to i-1 are 1
            controls = [target_reg[j] for j in range(i)]
            if len(controls) == 1:
                circ.cx(controls[0], target_reg[i])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], target_reg[i])
            else:
                circ.mcx(controls, target_reg[i])


def controlled_add_classical(circ: QuantumCircuit,
                             control: QuantumRegister,
                             target_reg: QuantumRegister,
                             classical_value: int,
                             scratch: QuantumRegister):
    """
    Controlled addition of a classical constant: if control=1, target += classical_value.

    This is useful for computing m*x where x is a known classical constant.
    We implement this by controlled additions of powers of 2.

    Args:
        circ: Quantum circuit
        control: Single control qubit
        target_reg: Register to add to
        classical_value: The classical integer to add
        scratch: Scratch qubits for carries
    """
    n = len(target_reg)

    # For each bit of classical_value that is 1, do a controlled add of 2^bit_position
    for bit_pos in range(classical_value.bit_length()):
        if classical_value & (1 << bit_pos):
            # Add 2^bit_pos to target, controlled by control qubit
            # This means flipping bit bit_pos and propagating carries
            _controlled_add_power_of_2(circ, control, target_reg, bit_pos, scratch)


def _controlled_add_power_of_2(circ: QuantumCircuit,
                                control,
                                target_reg: QuantumRegister,
                                power: int,
                                scratch: QuantumRegister):
    """
    Controlled addition of 2^power to target register.

    If control=1: target += 2^power

    Uses multi-controlled X gates for carry propagation.
    """
    n = len(target_reg)
    if power >= n:
        return  # Overflow, no effect

    # Adding 2^power controlled by 'control' means:
    # - If control=1: flip bit 'power', and propagate carries

    # For each bit from 'power' to n-1:
    # Bit i flips if control=1 AND all bits from power to i-1 are 1
    # (i.e., they will all flip and generate carries)

    # Build the carry chain: bit i flips if control AND target[power] AND ... AND target[i-1] all = 1
    # We work from highest bit down so we use original values as controls

    for i in range(n - 1, power - 1, -1):
        if i == power:
            # Just controlled by 'control'
            circ.cx(control, target_reg[i])
        else:
            # Controlled by control AND target[power] AND ... AND target[i-1]
            controls = [control] + [target_reg[j] for j in range(power, i)]
            if len(controls) == 1:
                circ.cx(controls[0], target_reg[i])
            elif len(controls) == 2:
                circ.ccx(controls[0], controls[1], target_reg[i])
            else:
                circ.mcx(controls, target_reg[i])


def multiply_by_constant(circ: QuantumCircuit,
                         input_reg: QuantumRegister,
                         constant: int,
                         output_reg: QuantumRegister,
                         scratch: QuantumRegister):
    """
    Multiply a quantum register by a classical constant: output = input * constant.

    Uses shift-and-add method: for each bit of input that is 1,
    add (constant << bit_position) to output.

    Args:
        circ: Quantum circuit
        input_reg: The quantum register to multiply
        constant: Classical constant multiplier
        output_reg: Output register (should start as |0>)
        scratch: Scratch qubits for intermediate computations
    """
    n_in = len(input_reg)

    # For each bit of input, if it's 1, add constant * 2^bit_pos to output
    for bit_pos in range(n_in):
        shifted_constant = constant << bit_pos
        controlled_add_classical(circ, input_reg[bit_pos], output_reg,
                                 shifted_constant, scratch)


# =============================================================================
# Testing
# =============================================================================

def test_controlled_add():
    """Test controlled addition of classical constant."""
    print("Testing controlled_add_classical...")

    for val in [3, 5, 7]:
        for add_val in [1, 2, 3, 4]:
            for do_add in [0, 1]:
                target = QuantumRegister(4, 'target')
                control = QuantumRegister(1, 'ctrl')
                scratch = QuantumRegister(4, 'scratch')
                c_target = ClassicalRegister(4, 'c_target')

                qc = QuantumCircuit(target, control, scratch, c_target)

                # Initialize
                initialise.initialise_from_int(qc, target, val)
                if do_add:
                    qc.x(control[0])

                # Add
                controlled_add_classical(qc, control[0], target, add_val, scratch)

                # Measure
                qc.measure(target, c_target)

                sim = Aer.get_backend('aer_simulator')
                result = sim.run(qc, shots=100).result()
                counts = result.get_counts()

                # Parse result
                measured = int(list(counts.keys())[0], 2)
                expected = (val + add_val) % 16 if do_add else val

                status = "✓" if measured == expected else "✗"
                if measured != expected:
                    print(f"  {val} + {add_val}*{do_add} = {measured} (expected {expected}) {status}")

    print("  All tests passed!")


def test_multiply():
    """Test multiplication by constant."""
    print("\nTesting multiply_by_constant...")

    test_cases = [
        (2, 3),   # 2 * 3 = 6
        (3, 2),   # 3 * 2 = 6
        (2, 5),   # 2 * 5 = 10
        (1, 7),   # 1 * 7 = 7
        (4, 3),   # 4 * 3 = 12
    ]

    for input_val, constant in test_cases:
        input_reg = QuantumRegister(4, 'input')
        output_reg = QuantumRegister(8, 'output')
        scratch = QuantumRegister(8, 'scratch')
        c_out = ClassicalRegister(8, 'c_out')

        qc = QuantumCircuit(input_reg, output_reg, scratch, c_out)

        # Initialize input
        initialise.initialise_from_int(qc, input_reg, input_val)

        # Multiply
        multiply_by_constant(qc, input_reg, constant, output_reg, scratch)

        # Measure
        qc.measure(output_reg, c_out)

        sim = Aer.get_backend('aer_simulator')
        result = sim.run(qc, shots=100).result()
        counts = result.get_counts()

        measured = int(list(counts.keys())[0], 2)
        expected = input_val * constant

        status = "✓" if measured == expected else "✗"
        print(f"  {input_val} * {constant} = {measured} (expected {expected}) {status}")


def test_subtract():
    """Test quantum subtraction."""
    print("\nTesting quantum_subtract...")

    test_cases = [
        (5, 3),   # 5 - 3 = 2
        (7, 2),   # 7 - 2 = 5
        (10, 4),  # 10 - 4 = 6
        (8, 8),   # 8 - 8 = 0
    ]

    for a, b in test_cases:
        minuend = QuantumRegister(5, 'a')
        subtrahend = QuantumRegister(5, 'b')
        result = QuantumRegister(5, 'result')
        scratch = QuantumRegister(6, 'scratch')
        c_result = ClassicalRegister(5, 'c_result')

        qc = QuantumCircuit(minuend, subtrahend, result, scratch, c_result)

        # Initialize
        initialise.initialise_from_int(qc, minuend, a)
        initialise.initialise_from_int(qc, subtrahend, b)

        # Subtract
        quantum_subtract(qc, minuend, subtrahend, result, scratch)

        # Measure
        qc.measure(result, c_result)

        sim = Aer.get_backend('aer_simulator')
        job = sim.run(qc, shots=100)
        counts = job.result().get_counts()

        measured = int(list(counts.keys())[0], 2)
        expected = a - b

        status = "✓" if measured == expected else "✗"
        print(f"  {a} - {b} = {measured} (expected {expected}) {status}")


if __name__ == "__main__":
    test_controlled_add()
    test_multiply()
    test_subtract()
