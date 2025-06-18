# QUBELLA: Quantum Neural Network Experiments

This repository contains experiments with Quantum Neural Networks (QNNs) using Qiskit, focusing on single-qubit rotations and parameter optimization.

## Project Structure

- `QUBELLA_V1/`: Single-parameter QNN experiments
  - `1QUBITQNN.py`: Implementation with one rotation parameter
- `QUBELLA_V1.1/`: Two-parameter QNN experiments
  - `1.1QUBITQNN.py`: Implementation with two rotation parameters

## Features

- Quantum circuit implementation using Qiskit
- Both EstimatorQNN and SamplerQNN implementations
- ADAM optimization for parameter learning
- Comprehensive visualization of quantum states and optimization landscapes
- 3D loss landscape visualization for multi-parameter optimization

## Requirements

- Python 3.8+
- Qiskit
- NumPy
- Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QUBELLA.git
cd QUBELLA

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the QNN experiments:

```bash
# Single parameter experiment
python QUBELLA_V1/1QUBITQNN.py

# Two parameter experiment
python QUBELLA_V1.1/1.1QUBITQNN.py
```

## License

MIT License 