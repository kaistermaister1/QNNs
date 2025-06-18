# Quantum Neural Networks (QNNs) Research Project

A comprehensive collection of Quantum Neural Network implementations and experiments using Qiskit, exploring various architectures and applications for quantum machine learning.

## 🎯 Project Overview

This repository contains practical implementations of QNNs across three main categories:

- **Manual QNN Implementations** - Custom quantum circuits and parameter optimization
- **Practical Demos** - Real-world classification problems and comparisons
- **Experimental Studies** - Performance analysis and architectural comparisons

## 📁 Project Structure

### `MANUAL QNNS/` - Custom QNN Implementations
- **QUBELLA V1**: Single-parameter quantum neural network with RY rotations
- **QUBELLA V1.1**: Two-parameter QNN with enhanced optimization landscapes
- Features EstimatorQNN and SamplerQNN implementations with ADAM optimization

### `DEMOS/` - Applied QNN Examples

#### `IRIS/` - Multi-Class Classification
Comprehensive comparison of 6 different QNN architectures for IRIS flower classification:
- VQC models with ZZFeatureMap and custom feature maps (2-4 qubits)
- Siamese-like QNN architectures for binary pair classification
- Custom vs. standard Qiskit ansatz comparisons

#### `LINE_CLASSIFICATION/` - Binary Classification
Systematic study of QNN performance on 2D point classification:
- **1-qubit models**: Angle embedding (~73% accuracy) and amplitude embedding
- **2-qubit models**: Custom architecture achieving ~88% accuracy
- **Default Qiskit**: Standard ZZFeatureMap + RealAmplitudes (~52% accuracy)

#### `Estimator_Sampler_demo.ipynb`
Educational notebook demonstrating the difference between Qiskit's Estimator and Sampler primitives.

## 🔧 Requirements

Install dependencies from `MANUAL QNNS/requirements.txt`:

```bash
pip install -r "MANUAL QNNS/requirements.txt"
```

**Key Dependencies:**
- Qiskit 1.4.2
- Qiskit Machine Learning 0.8.2
- NumPy, Matplotlib, Scikit-learn
- Pandas, Seaborn (for visualizations)

## 🚀 Quick Start

### Run IRIS Classification Comparison
```bash
python "DEMOS/IRIS/iris_comparison.py"
```

### Test Line Classification Models
```bash
# Run comprehensive comparison study
python "DEMOS/LINE_CLASSIFICATION/qnn_comparison_study.py"
```

### Try Manual QNN Implementation
```bash
# Single parameter QNN
python "MANUAL QNNS/QUBELLA_V1/1QUBITQNN.py"

# Two parameter QNN with 3D landscapes
python "MANUAL QNNS/QUBELLA_V1.1/1.1QUBITQNN.py"
```

## 📊 Key Findings

- **Fewer qubits can outperform more qubits** when feature encoding is properly designed
- **Custom feature maps** suited to specific data structures perform better than generic approaches
- **Direct angle encoding** often superior to correlation-based feature maps for independent features
- **Manual QNN implementations** provide fine-grained control over circuit design and optimization

## 📈 Performance Highlights

- **Line Classification**: Custom 2-qubit model achieved 88% accuracy vs 52% for standard Qiskit approach
- **IRIS Classification**: Multiple architectures tested with comprehensive statistical analysis
- **QUBELLA**: Detailed optimization landscapes and quantum state visualizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 