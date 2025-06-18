# Quantum Neural Network (QNN) Experiments

## Overview

This folder contains experiments with different Quantum Neural Network architectures for binary classification of 2D points.

## Models

### `qiskit_qnn_DEFAULT.ipynb` - Official Qiskit Tutorial
- **Source**: [Qiskit Machine Learning Tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html)
- **Architecture**: 2-qubit ZZFeatureMap + RealAmplitudes ansatz
- **Performance**: ~52% accuracy

### Custom Modifications

The original tutorial used a ZZFeatureMap designed to capture feature correlations, which doesn't make sense for our dataset of independent random points in a 2D box. The following modifications were made:

#### `qiskit_qnn_2QUBITS.ipynb` - Custom 2-Qubit Model
- **Architecture**: 2-qubit custom angle embedding (RY gates) + custom ansatz
- **Improvement**: Direct angle encoding without unnecessary feature correlations
- **Performance**: ~88% accuracy

#### `qiskit_qnn_1QUBIT.ipynb` - 1-Qubit Models
Contains two approaches:

1. **Angle Embedding**: 1-qubit with RY + RZ rotations
   - Maps 2D coordinates directly to rotation angles
   - **Performance**: ~73% accuracy

2. **Amplitude Embedding**: 1-qubit with single RY rotation
   - Preprocesses 2D coordinates into polar angles
   - **Performance**: ~73% accuracy

## Key Insights

- **Better performance with fewer qubits**: 1-qubit models outperformed the 2-qubit ZZFeatureMap
- **Feature encoding matters**: Custom encodings suited to the data structure perform better than generic feature maps

## Comparison Study

Run `qnn_comparison_study.py` to perform a comprehensive statistical comparison of all 4 models across 100 random datasets. 