from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import initialise
from qiskit_aer import Aer

def increment_by_one_no_control(
    circ: QuantumCircuit,
    target_reg: QuantumRegister,
    carry_flag: QuantumRegister,
    scratch: QuantumRegister,
    name_prefix="inc1"
):
    """
    Increment 'target_reg' by 1 (mod 2^len(target_reg)),
    *provided* that 'carry_flag' starts as |1>.
    If carry_flag is |0>, this adds 0.

    Uses a ripple-carry approach with scratch qubits to store intermediate carries.

    - 'target_reg[i]' is bit i (LSB = i=0).
    - 'carry_flag' is a single-qubit register (the initial carry-in).
    - 'scratch' stores intermediate carries; needs len(target_reg) qubits.
    - At the end, 'carry_flag' is unchanged, scratch is back to |0>.
    """
    n = len(target_reg)
    if len(scratch) < n:
        raise ValueError(f"Need at least {len(target_reg)=} scratch qubits for naive ripple increment")

    # scratch[i] will hold the carry INTO bit i+1
    # scratch[0] = carry_flag AND target[0] (carry into bit 1)
    # scratch[i] = scratch[i-1] AND target[i] (carry into bit i+1)

    # Forward pass: compute carries
    # carry into bit 1 = carry_flag AND target[0]
    circ.ccx(carry_flag, target_reg[0], scratch[0])

    # carry into bit i+1 = carry_into_bit_i AND target[i]
    for i in range(1, n - 1):
        circ.ccx(scratch[i-1], target_reg[i], scratch[i])

    # Apply sums (XOR each bit with its carry-in)
    # bit 0: XOR with carry_flag
    circ.cx(carry_flag, target_reg[0])

    # bit i: XOR with scratch[i-1] (carry into bit i)
    for i in range(1, n):
        circ.cx(scratch[i-1], target_reg[i])

    # Backward pass: uncompute carries (in reverse order)
    for i in range(n - 2, 0, -1):
        # To uncompute scratch[i], we need the ORIGINAL values of scratch[i-1] and target[i]
        # scratch[i-1] is unchanged (we haven't touched it yet in uncompute)
        # target[i] has been flipped if scratch[i-1] was 1
        # So: current_target[i] = original_target[i] XOR scratch[i-1]
        # original_target[i] = current_target[i] XOR scratch[i-1]

        # Undo the sum first to get original target[i]
        circ.cx(scratch[i-1], target_reg[i])
        # Now uncompute scratch[i]
        circ.ccx(scratch[i-1], target_reg[i], scratch[i])
        # Redo the sum
        circ.cx(scratch[i-1], target_reg[i])

    # Uncompute scratch[0]
    # target[0] has been flipped if carry_flag was 1
    # Undo sum, uncompute, redo sum
    circ.cx(carry_flag, target_reg[0])
    circ.ccx(carry_flag, target_reg[0], scratch[0])
    circ.cx(carry_flag, target_reg[0])


def test_suite(starting_number, carry=0):
    target = QuantumRegister(4, "target")
    target_c = ClassicalRegister(4, 'target_c')

    carry_flag = QuantumRegister(1, "carry_flag")
    carry_flag_c = ClassicalRegister(1, "carry_flag_c")

    scratch = QuantumRegister(4, "scratch")
    scratch_c = ClassicalRegister(4, 'scratch_c')
    
    qc = QuantumCircuit(target, target_c, carry_flag, carry_flag_c, scratch, scratch_c)
    initialise.initialise_from_int(qc, target, starting_number)
    initialise.initialise_from_int(qc, carry_flag, carry)
    increment_by_one_no_control(qc, target, carry_flag, scratch)
    qc.barrier()
    qc.measure(target, target_c)

    sim = Aer.get_backend('aer_simulator')
    job = sim.run(qc)
    result = job.result()
    counts = result.get_counts()
    print(f"{counts=}")

if __name__ == '__main__':
    test_suite(7)
    test_suite(7,1)
