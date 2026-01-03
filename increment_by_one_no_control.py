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
    
    Implementation: classical ripple-carry for "target_reg += carry_flag".
    
    - 'target_reg[i]' is bit i (LSB = i=0).
    - 'carry_flag' is a single-qubit register. 
      If it starts as |1>, we add 1. If it's |0>, we add 0.
    - 'scratch' is a set of ancillas, at least as many as len(target_reg) 
      (one scratch qubit per bit), so we can do Toffoli without overwriting info. 
    - At the end, 'carry_flag' holds the final carry-out. 
      The 'scratch' qubits are uncomputed back to |0>. 
    """

    n = len(target_reg)
    if len(scratch) < n:
        raise ValueError(f"Need at least {len(target_reg)=} scratch qubits for naive ripple increment")

    for i in range(n):
        bit_i = target_reg[i]
        tmp_i = scratch[i]
        # We'll do a standard "add carry to bit_i" step:
        #
        # new_bit_i = bit_i XOR carry_flag
        # new_carry = bit_i AND carry_flag
        #
        # Implementation with gates:
        #   tmp_i = bit_i AND carry_flag   (Toffoli)
        #   bit_i = bit_i XOR carry_flag   (CNOT)
        #   carry_flag = carry_flag XOR tmp_i   (CNOT)
        # So after these 3 gates:
        #   bit_i becomes old_bit_i ^ old_carry_flag
        #   carry_flag becomes old_carry_flag ^ (old_bit_i & old_carry_flag) = old_bit_i & old_carry_flag
        #   tmp_i is still old_bit_i & old_carry_flag
        # We'll uncompute tmp_i so it ends up back in |0>.
        
        # tmp_i = bit_i AND carry_flag
        circ.ccx(bit_i, carry_flag, tmp_i)  # Toffoli
        # bit_i = bit_i XOR carry_flag
        circ.cx(carry_flag, bit_i)
        # carry_flag = carry_flag XOR tmp_i
        circ.cx(tmp_i, carry_flag)
        
        # Now we uncompute tmp_i = old_bit_i & old_carry_flag
        circ.ccx(bit_i, carry_flag, tmp_i)  # same Toffoli again reverts tmp_i to |0>
        # Done. The new carry_flag is (old_bit_i & old_carry_flag).


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
