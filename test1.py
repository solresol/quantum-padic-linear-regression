import qiskit
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

fig = circ.draw("mpl")
fig.savefig("test1.png")

simulator = AerSimulator()
result = simulator.run(circ).result()
counts = result.get_counts(circ)

fig = plot_histogram(counts, title='3-qubit thing')
fig.savefig("test1-hist.png")
## Construct an ideal simulator with SamplerV2
#sampler = SamplerV2()
#job = sampler.run([circ], shots=128)

# Perform an ideal simulation
#result_ideal = job.result()
#counts_ideal = result_ideal[0].data.meas.get_counts()
#print('Counts(ideal):', counts_ideal)
