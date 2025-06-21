#!/usr/bin/env python3
"""
Circuit Visualizer for IRIS QNN Models
======================================

This script extracts and visualizes the quantum circuits from all 6 models
used in the IRIS comparison study.
"""

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
import os

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

print("🎨 Generating circuit visualizations for all 6 IRIS QNN models")
print("=" * 60)

# ============================================================================
# MODEL 1: VQC with ZZFeatureMap + RealAmplitudes (4 features)
# ============================================================================
print("📊 Model 1: VQC (4 Features, ZZFeatureMap + RealAmplitudes)")

# Feature map
feature_map_1 = ZZFeatureMap(feature_dimension=4, reps=1)
# Ansatz
ansatz_1 = RealAmplitudes(num_qubits=4, reps=3)
# Combined circuit
circuit_1 = QuantumCircuit(4)
circuit_1 = circuit_1.compose(feature_map_1)
circuit_1 = circuit_1.compose(ansatz_1)

# Draw and save
fig = circuit_1.draw(output='mpl', style='clifford')
plt.title('Model 1: VQC with ZZFeatureMap + RealAmplitudes (4 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/demoVQCnoncondensed.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 2: VQC with ZZFeatureMap + EfficientSU2 (2 features)
# ============================================================================
print("📊 Model 2: VQC (2 Features, ZZFeatureMap + EfficientSU2)")

# Feature map
feature_map_2 = ZZFeatureMap(feature_dimension=2, reps=1)
# Ansatz
ansatz_2 = EfficientSU2(num_qubits=2, reps=3)
# Combined circuit
circuit_2 = QuantumCircuit(2)
circuit_2 = circuit_2.compose(feature_map_2)
circuit_2 = circuit_2.compose(ansatz_2)

# Draw and save
fig = circuit_2.draw(output='mpl', style='clifford')
plt.title('Model 2: VQC with ZZFeatureMap + EfficientSU2 (2 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/demoVQCcondensed.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 3: Siamese-like QNN (4-feature pairs)
# ============================================================================
print("📊 Model 3: Siamese-like QNN (4-feature pairs)")

# Define circuit (copied from iris_comparison.py)
circuit_3 = QuantumCircuit(4)
input_params3 = [Parameter(f"input{i}") for i in range(8)]
weight_params3 = [Parameter(f"weight{i}") for i in range(8)]

# Feature map part
circuit_3.ry(input_params3[0], 0)
circuit_3.rz(input_params3[1], 0)
circuit_3.ry(input_params3[2], 1)
circuit_3.rz(input_params3[3], 1)
circuit_3.ry(input_params3[4], 2)
circuit_3.rz(input_params3[5], 2)
circuit_3.ry(input_params3[6], 3)
circuit_3.rz(input_params3[7], 3)
circuit_3.cx(0, 2)
circuit_3.cx(1, 3)
circuit_3.barrier()

# Ansatz part
circuit_3.ry(weight_params3[0], 0)
circuit_3.rz(weight_params3[1], 1)
circuit_3.ry(weight_params3[2], 2)
circuit_3.rz(weight_params3[3], 3)
circuit_3.rz(weight_params3[4], 0)
circuit_3.ry(weight_params3[5], 1)
circuit_3.rz(weight_params3[6], 2)
circuit_3.ry(weight_params3[7], 3)

# Draw and save
fig = circuit_3.draw(output='mpl', style='clifford')
plt.title('Model 3: Siamese-like QNN (4-feature pairs, 4 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/siamesenoncondensed.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 4: Siamese-like QNN (2-feature pairs)
# ============================================================================
print("📊 Model 4: Siamese-like QNN (2-feature pairs)")

# Define circuit (copied from iris_comparison.py)
circuit_4 = QuantumCircuit(4)
input_params4 = [Parameter(f"input{i}") for i in range(4)]
weight_params4 = [Parameter(f"weight{i}") for i in range(8)]

# Feature map part
circuit_4.ry(input_params4[0], 0)
circuit_4.ry(input_params4[1], 1)
circuit_4.ry(input_params4[2], 2)
circuit_4.ry(input_params4[3], 3)
circuit_4.cx(0, 2)
circuit_4.cx(1, 3)
circuit_4.barrier()

# Ansatz part
circuit_4.ry(weight_params4[0], 0)
circuit_4.ry(weight_params4[1], 1)
circuit_4.ry(weight_params4[2], 2)
circuit_4.ry(weight_params4[3], 3)
circuit_4.rz(weight_params4[4], 0)
circuit_4.rz(weight_params4[5], 1)
circuit_4.rz(weight_params4[6], 2)
circuit_4.rz(weight_params4[7], 3)

# Draw and save
fig = circuit_4.draw(output='mpl', style='clifford')
plt.title('Model 4: Siamese-like QNN (2-feature pairs, 4 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/siamesecondensed.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 5: VQC with Custom Feature Map + Custom Ansatz (4 features)
# ============================================================================
print("📊 Model 5: VQC (4 Features, Custom Feature Map + Custom Ansatz)")

# Custom feature map
feature_map_5 = QuantumCircuit(4)
input_params5 = [Parameter(f"input{i}") for i in range(4)]
feature_map_5.ry(input_params5[0], 0)
feature_map_5.ry(input_params5[1], 1)
feature_map_5.ry(input_params5[2], 2)
feature_map_5.ry(input_params5[3], 3)

# Custom ansatz
ansatz_5 = QuantumCircuit(4)
weight_params5 = [Parameter(f"weight{i}") for i in range(8)]
ansatz_5.ry(weight_params5[0], 0)
ansatz_5.ry(weight_params5[1], 1)
ansatz_5.ry(weight_params5[2], 2)
ansatz_5.ry(weight_params5[3], 3)
ansatz_5.rz(weight_params5[4], 0)
ansatz_5.rz(weight_params5[5], 1)
ansatz_5.rz(weight_params5[6], 2)
ansatz_5.rz(weight_params5[7], 3)

# Combined circuit
circuit_5 = QuantumCircuit(4)
circuit_5 = circuit_5.compose(feature_map_5)
circuit_5 = circuit_5.compose(ansatz_5)

# Draw and save
fig = circuit_5.draw(output='mpl', style='clifford')
plt.title('Model 5: VQC with Custom Feature Map + Custom Ansatz (4 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/irisnoncondensed.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 6: VQC with custom feature map and ansatz (2 features)
# ============================================================================
print("📊 Model 6: VQC (2 Features, Custom Feature Map + Custom Ansatz)")

# Custom feature map
feature_map_6 = QuantumCircuit(2)
input_params6 = [Parameter(f"input{i}") for i in range(2)]
feature_map_6.ry(input_params6[0], 0)
feature_map_6.ry(input_params6[1], 1)

# Custom ansatz
ansatz_6 = QuantumCircuit(2)
weight_params6 = [Parameter(f"weight{i}") for i in range(4)]
ansatz_6.ry(weight_params6[0], 0)
ansatz_6.rz(weight_params6[1], 0)
ansatz_6.ry(weight_params6[2], 1)
ansatz_6.rz(weight_params6[3], 1)

# Combined circuit
circuit_6 = QuantumCircuit(2)
circuit_6 = circuit_6.compose(feature_map_6)
circuit_6 = circuit_6.compose(ansatz_6)

# Draw and save
fig = circuit_6.draw(output='mpl', style='clifford')
plt.title('Model 6: VQC with Custom Feature Map + Custom Ansatz (2 qubits)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/iriscondensed.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✨ All circuit visualizations completed!")
print("📁 Saved 6 circuit diagrams to DEMOS/IRIS/plots/:")