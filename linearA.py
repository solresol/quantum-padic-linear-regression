#!/usr/bin/env python3

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import QuantumRegister, ClassicalRegister

n = 3  # number of qubits to encode x (values 0..7)
x_reg = QuantumRegister(n, name='x')
circ = QuantumCircuit(x_reg)

# Create uniform superposition of 0..(2^n - 1)
circ.h(x_reg)

