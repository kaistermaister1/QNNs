# SHORA - Quantum Neural Networks for Integer Factorization

A collection of quantum neural network experiments inspired by Shor's algorithm, exploring QNN-based approaches to learning integer factorization patterns.

## 🎯 Project Goal

Train quantum neural networks to predict the **lowest prime factor** of integers 2-15, effectively learning to factor numbers through supervised learning rather than the traditional quantum Fourier transform approach.

## 📁 File Structure

### Core Experiments
- **`babyshora.py`** - Original 2-qubit proof-of-concept
- **`babyshora2.1.py`** - Enhanced 2-qubit version with improved ansatz  
- **`babyshora2.2.py`** - Advanced 2-qubit with 6 parameters and comprehensive plotting
- **`babyshorai.py`** - **Recommended** 4-qubit version with proper cross-entropy loss

### Supporting Files
- **`plots/`** - Generated circuit diagrams and training results
- **`requirements.txt`** - Python dependencies

## 🔬 Technical Evolution

### Version Progression

| Version | Qubits | Parameters | Loss Function | Key Features |
|---------|--------|------------|---------------|--------------|
| `babyshora.py` | 2 | 4 | Hamming Distance | Mode-based prediction |
| `babyshora2.1.py` | 2 | 4 | Hamming Distance | Improved feature map |
| `babyshora2.2.py` | 2 | 6 | **Squared Error** | Probability thresholding, enhanced plots |
| `babyshorai.py` | **4** | ~16 | **Cross-Entropy** | Proper gradients, one-hot targets |

### Key Improvements in `babyshorai.py`
- ✅ **4-qubit circuit** with RealAmplitudes ansatz (sufficient expressivity)
- ✅ **Cross-entropy loss** with meaningful gradients (vs. flat plateau loss)
- ✅ **One-hot target encoding** over 16 classes
- ✅ **Angle encoding** feature map: `Ry(π * bit)` per qubit
- ✅ **ADAM/COBYLA optimizer choice** with proper API usage

## 🚀 Quick Start

### Run the Best Version
```bash
python babyshorai.py
```

### Compare Older Approaches
```bash
# Probability thresholding approach (will struggle to converge)
python babyshora2.2.py

# Switch between ADAM/COBYLA in the file:
OPTIMIZER_CHOICE = "ADAM"  # or "COBYLA"
```

## 📊 Expected Results

### `babyshorai.py` (Designed by AI)
- **Training Time**: 30-90 seconds
- **Expected Accuracy**: ~5% (depending on convergence)

### `babyshora2.2.py` (Best performing)
- **Training Time**: 60-180 seconds  
- **Expected Accuracy**: 10-30%

## 🎯 Factorization Mapping

The QNN learns to map:
```
Input: 4-bit binary → Output: Lowest prime factor

Examples:
4  (0100₂) → 2
6  (0110₂) → 2  
9  (1001₂) → 3
15 (1111₂) → 3
```

Once the QNN predicts factor `p`, the complete factorization is `(p, n//p)`.

## 📈 Performance Visualization

The plots show:
- **Training loss curves** (convergence behavior)
- **Prediction accuracy** by integer (which numbers are hardest)
- **Predicted vs actual factors** scatter plot
- **Side-by-side factor comparison** for all integers

## 🔧 Dependencies

```bash
pip install qiskit qiskit-algorithms qiskit-machine-learning numpy matplotlib
```

See `requirements.txt` for exact versions.

## 🎓 Educational Value

This project demonstrates:
- **Quantum machine learning** fundamentals
- **Loss function design** importance for QNN training
- **Circuit expressivity** vs. optimization landscape trade-offs
- **Feature encoding** strategies for quantum circuits
- **Comparison of optimization methods** (ADAM vs COBYLA)

Perfect for understanding why many QNN approaches fail and how to design quantum circuits that actually learn! 