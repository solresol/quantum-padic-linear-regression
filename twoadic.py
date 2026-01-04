#!/usr/bin/env python3

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from increment_by_one_no_control import increment_by_one_no_control
import initialise

def increment_by_one_if(
    circ: QuantumCircuit,
    target_reg: QuantumRegister,
    controls: list,         # list of control qubits that must ALL be 1
    carry_flag: QuantumRegister,
    scratch: QuantumRegister,
    name_prefix="inc_if"
):
    """
    If *all* qubits in 'controls' are 1, increment 'target_reg' by 1, 
    using the 'carry_flag' approach from increment_by_one_no_control.
    
    Steps:
    - Build an ancilla "all_ones" that is 1 iff all controls are 1.
    - Set carry_flag = all_ones (so if all_ones=1, we effectively do +1, else +0).
    - Do the ripple-carry increment.
    - Uncompute 'all_ones' so ancillas end in |0>.
    """
    if len(controls) == 0:
        # If no controls, then this is unconditional increment. Just set carry_flag=1 and increment.
        circ.x(carry_flag)  # set carry_flag=1
        increment_by_one_no_control(circ, target_reg, carry_flag, scratch, name_prefix=name_prefix)
        circ.x(carry_flag)  # restore it to 0
        return
    
    # 1) "all_ones" ancilla:
    all_ones = scratch[0]
    # set all_ones = AND of all controls
    # We'll do a chain of Toffoli or MCX. Qiskit has MCXGate if you like:
    # but let's do a naive approach with a loop of ccx.  For 2 controls it's straightforward.
    
    # Initialize all_ones to 1
    circ.x(all_ones)
    # Now flip it to 0 if any control is 0
    circ.mcx(controls, all_ones)  # multi-controlled X will flip 'all_ones' back to 0 
                                  # if *all* controls are 1, then we do an extra flip => ends at 1
                                  # Actually Qiskit's MCX toggles the target if all controls=1. 
                                  # We started all_ones=1, so if all controls=1, it toggles to 0 => which is the opposite of what we want. 
                                  # Let's fix that logic below.
    circ.x(all_ones)             # re-flip so that if all_ones was toggled, we get 1, else 0.
    
    # Now all_ones = 1 iff all controls=1, else 0.
    
    # 2) Move all_ones -> carry_flag
    circ.cx(all_ones, carry_flag)
    
    # 3) increment target_reg by carry_flag
    increment_by_one_no_control(circ, target_reg, carry_flag, scratch[1:], name_prefix=name_prefix)
    
    # 4) uncompute all_ones from carry_flag
    circ.cx(all_ones, carry_flag)
    
    # 5) reset all_ones to |0>
    # We set it back to 1 if it was toggled. Let's do the same steps in reverse:
    circ.x(all_ones)
    circ.mcx(controls, all_ones)
    circ.x(all_ones)


def stop_if_bit_is_1(
    circ: QuantumCircuit,
    still_zero,  # single qubit
    bit_is_1,    # single qubit (the actual bit from diff_reg)
    anc,         # list/register of ancillas - we use anc[0] as tmp
    name_prefix="stop_if_1"
):
    """
    still_zero_{new} = still_zero_{old} AND (NOT bit_is_1)
    i.e. set 'still_zero' to 0 if the bit is 1 and 'still_zero' was 1,
    otherwise leave it as is.

    Simple implementation: uses anc[0] as tmp but does NOT uncompute it.
    The caller is responsible for ensuring tmp starts at 0 and will stay
    dirty after this call (contains old_still_zero AND bit).

    For proper uncomputation, the caller should reset tmp before reuse
    or allocate separate tmps per call.
    """
    tmp = anc[0]
    # tmp = still_zero AND bit_is_1
    circ.ccx(still_zero, bit_is_1, tmp)
    # flip still_zero if tmp=1
    circ.cx(tmp, still_zero)
    # NOTE: tmp is left dirty (contains old_still_zero AND bit)


def count_trailing_zeros_inplace(
    circ: QuantumCircuit,
    diff_reg: QuantumRegister,
    tz_reg: QuantumRegister,
    anc_reg: QuantumRegister,
    name_prefix="ctz"
):
    """
    Purely unitary subcircuit that computes the number of trailing zeros in `diff_reg`
    (looking from LSB to MSB) and stores that count in `tz_reg`.
    - `tz_reg` is incremented by 1 for each zero-bit we see (as long as `still_zero` is 1).
    - Once we encounter a 1-bit, we set `still_zero=0` and skip increments for subsequent bits.
    
    Requirements/Assumptions:
      - `diff_reg[0]` is the LSB.
      - `tz_reg` is large enough to hold values up to len(diff_reg).
      - `anc_reg` should have enough qubits for multi-controlled increments and small subroutines.
        Let's assume anc_reg has at least (1 + |tz_reg| + 1) for safety:
          * 1 qubit for `still_zero`
          * ~|tz_reg| for the increment carry scratch
          * 1 more for the stop_if_bit_is_1 scratch
      - Both `tz_reg` and `still_zero` start in |0>. We'll set `still_zero` to |1> at the start.
    
    After this subcircuit:
      - `tz_reg` contains the integer number of trailing zeros in `diff_reg`.
      - `still_zero` ends in 0 if we found a '1' at some point, or 1 if `diff_reg` was all zero.
      - The rest of anc_reg is uncomputed back to |0>.
    """
    n = len(diff_reg)
    r = len(tz_reg)
    if (1 << r) < n:
        raise ValueError("tz_reg is too small to hold a count up to n.")

    # Allocate ancilla slices:
    # - 1 qubit for still_zero
    # - 2 + len(tz_reg) for increment scratch
    # - 1 for all_controls
    # - n qubits for stop_if_bit_is_1 tmp (one per bit, since tmp stays dirty)
    still_zero = anc_reg[0]
    scratch_for_increment = anc_reg[1 : 1 + 2 + len(tz_reg)]
    all_controls_idx = 1 + 2 + len(tz_reg)
    all_controls = anc_reg[all_controls_idx]
    stop_tmp_start = all_controls_idx + 1  # tmp qubits for stop_if_bit_is_1

    min_anc_needed = stop_tmp_start + n
    if len(anc_reg) < min_anc_needed:
        raise ValueError(f"count_trailing_zeros needs at least {min_anc_needed} ancilla qubits, got {len(anc_reg)}")

    # 1) Set still_zero = 1
    circ.x(still_zero)

    # 2) For each bit i in diff_reg:
    for i in range(n):
        # If bit is 0 AND still_zero=1 => increment tz_reg
        circ.x(diff_reg[i])  # flip so 0->1

        # all_controls = still_zero AND flipped_bit
        circ.ccx(still_zero, diff_reg[i], all_controls)

        # increment tz_reg if all_controls=1
        increment_by_one_if(
            circ,
            tz_reg,
            controls=[all_controls],
            carry_flag=scratch_for_increment[0],
            scratch=scratch_for_increment[1:],
            name_prefix=f"{name_prefix}_inc_bit{i}"
        )

        # uncompute all_controls
        circ.ccx(still_zero, diff_reg[i], all_controls)

        # restore diff_reg[i]
        circ.x(diff_reg[i])

        # If bit is 1, set still_zero=0
        # Use a UNIQUE tmp for each bit (tmp stays dirty after stop_if_bit_is_1)
        stop_tmp = [anc_reg[stop_tmp_start + i]]
        stop_if_bit_is_1(
            circ,
            still_zero,
            diff_reg[i],
            stop_tmp,
            name_prefix=f"{name_prefix}_stop_bit{i}"
        )

        circ.barrier()


#
#  DEMO: Putting it all together in a small circuit
#
if __name__ == "__main__":
    from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
    from qiskit_aer import Aer

    #number_to_analyse = 12
    number_to_analyse = 8    
    number_of_bits_required = initialise.number_of_bits_required(number_to_analyse)
    print(f"We need {number_of_bits_required} bits")
    # Example: we have a 4-bit diff_reg, and we want to store the trailing-zero count in a 3-bit trailing_zero_count_register
    tz_n = 3
    
    diff_reg = QuantumRegister(number_of_bits_required, 'diff')
    trailing_zero_count_register   = QuantumRegister(tz_n,   'tz')
    anc_reg  = QuantumRegister(1 + 2 + tz_n + 2, 'anc')
    # Explanation: 1 qubit for still_zero + (2 + tz_n) for increment scratch + 2 for other logic
    # scratch_for_increment needs: 1 (carry_flag) + 1 (all_ones) + tz_n (ripple increment)
    
    c_diff = ClassicalRegister(number_of_bits_required, 'c_diff')
    c_tz   = ClassicalRegister(tz_n,   'c_tz')
    
    qc = QuantumCircuit(diff_reg, trailing_zero_count_register, anc_reg, c_diff, c_tz)
    
    # 1) Load some test value into diff_reg. Let's pick e.g. 0b01100 => decimal 12 => trailing zeros=2
    #    But we only have 4 bits, so let's pick something that fits. E.g. 0b0100 => decimal 4 => 2 trailing zeros
    #    We'll do it by hand: set diff_reg=4 => bits are [LSB..MSB] => 0100
    #    That means diff_reg[0]=0, diff_reg[1]=0, diff_reg[2]=1, diff_reg[3]=0
    #    We'll just do X on bit 2.

    initialise.initialise_from_int(qc, diff_reg, number_to_analyse)
    #for i in range(number_of_bits_required):
    #    if number_to_analyse & (1 << i):
    #        print("I will flip",i)
    #        qc.x(diff_reg[i])
    
    # 2) Run the trailing-zero counter
    count_trailing_zeros_inplace(qc, diff_reg, trailing_zero_count_register, anc_reg, name_prefix="ctz_demo")
    
    # 3) Measure
    qc.barrier()
    qc.measure(diff_reg, c_diff)
    qc.measure(trailing_zero_count_register, c_tz)
    
    # 4) Execute
    sim = Aer.get_backend('aer_simulator')
    job = sim.run(qc)
    result = job.result()
    counts = result.get_counts()
    print("RESULT:", counts)
    
    # If diff_reg was 4 (binary 0100), we expect trailing_zero_count_register to end up as 2 (binary 010).
    # So we hope to see something like "diff=0100 tz=010" in the output.

    print("\nFULL CIRCUIT:\n")
    full_circuit_figure = qc.draw("mpl")
    full_circuit_figure.savefig("twoadic.png")
