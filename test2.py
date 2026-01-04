# Built-in modules
import math


# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.visualization import plot_distribution

from qiskit_aer import AerSimulator


def grover_oracle(marked_states):
    """Build a Grover oracle for multiple marked states

    Here we assume all input marked states have the same number of bits

    Parameters:
        marked_states (str or list): Marked states of oracle

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        qc.x(zero_inds)
        # Use mcz (multi-controlled Z) instead of deprecated MCMT
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            # MCZ = H on target, MCX, H on target
            target_qubit = num_qubits - 1
            control_qubits = list(range(num_qubits - 1))
            qc.h(target_qubit)
            qc.mcx(control_qubits, target_qubit)
            qc.h(target_qubit)
        qc.x(zero_inds)
    return qc

def grover_diffusion(num_qubits):
    """Create Grover diffusion operator."""
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    # Multi-controlled Z
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        target = num_qubits - 1
        controls = list(range(num_qubits - 1))
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


marked_states = ["011", "100"]

oracle = grover_oracle(marked_states)
fig = oracle.draw(output="mpl", style="iqp")
fig.savefig("test2.png")

num_qubits = len(marked_states[0])
diffusion = grover_diffusion(num_qubits)
fig = diffusion.draw(output="mpl", style="iqp")
fig.savefig("test2b.png")

# Calculate optimal iterations
n_solutions = len(marked_states)
n_total = 2**num_qubits
optimal_num_iterations = max(1, math.floor(
    math.pi / (4 * math.asin(math.sqrt(n_solutions / n_total)))
))

# Build full Grover circuit
qc = QuantumCircuit(num_qubits)
# Create even superposition of all basis states
qc.h(range(num_qubits))
# Apply Grover iterations
for _ in range(optimal_num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)
# Measure all qubits
qc.measure_all()
fig = qc.draw(output="mpl", style="iqp")
fig.savefig("test2c.png")

# Run on local simulator
simulator = AerSimulator()
result = simulator.run(qc).result()
dist = result.get_counts(qc)

fig = plot_distribution(dist)
fig.savefig("test2d.png")

print(f"Marked states: {marked_states}")
print(f"Measurement results: {dist}")
