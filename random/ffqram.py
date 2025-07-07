from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
import matplotlib.pyplot as plt

qreg_q = QuantumRegister(3, 'q')

circuit = QuantumCircuit(qreg_q)

circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.x(qreg_q[0])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.x(qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.x(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.x(qreg_q[1])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
# @columns [0,0,1,1,1,3,5,5,5,6,7,8,9,9,9,10,11,12,13,13,13,14,14,15,16,16,17,17,17]

# Display circuit with aligned lines
fig = circuit.draw(output='mpl', style='iqp', fold=-1, justify='none', plot_barriers=False)
plt.tight_layout()
plt.show()

# Optional: Print circuit information
print("Circuit depth:", circuit.depth())
print("Number of qubits:", circuit.num_qubits)
print("Number of operations:", len(circuit.data))