import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from tqdm import tqdm

import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs("plots", exist_ok=True)


# Configuration
NUM_TRIALS = 10
MAX_ITER = 80
NUM_FEATURES = 3  # Number of features to select using FS1
NUM_TRAIN_SAMPLES = 180  # Number of samples to use for training (remaining used for testing)
# algorithm_globals.random_seed = 123
# np.random.seed(algorithm_globals.random_seed)


# --- Data Loading and Preprocessing ---

def load_and_prep_data(num_features=None, feature_selection_method='FS1', num_train_samples=None):
    """
    Load and preprocess the HTRU_2 dataset with feature selection.
    
    Parameters:
    -----------
    num_features : int, optional
        Number of features to select. If None, uses all features.
    feature_selection_method : str
        Feature selection method to use. Currently supports 'FS1' (SelectKBest with f_classif).
    num_train_samples : int, optional
        Number of samples to use for training. Remaining samples will be used for testing.
        If None, uses 80/20 split on all samples.
    
    Returns:
    --------
    train_features : np.ndarray
        Training features (selected and scaled)
    test_features : np.ndarray  
        Testing features (selected and scaled)
    train_labels : np.ndarray
        Training labels
    test_labels : np.ndarray
        Testing labels
    """
    # Load the HTRU_2 dataset
    df = pd.read_csv('HTRU_2.csv', header=None)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column as labels
    
    total_samples = len(X)
    
    # Split into train/test based on num_train_samples
    if num_train_samples is not None and num_train_samples < total_samples:
        # Use stratified sampling to get training set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=num_train_samples, random_state=42, stratify=y
        )
        print(f"📊 Training: {X_train.shape[0]} samples ({np.bincount(y_train)[0]} class 0, {np.bincount(y_train)[1]} class 1)")
        print(f"📊 Testing: {X_test.shape[0]} samples ({np.bincount(y_test)[0]} class 0, {np.bincount(y_test)[1]} class 1)")
    else:
        # Use 80/20 split on all samples
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42, stratify=y
        )
        print(f"📊 Using 80/20 split on all {total_samples} samples")
        print(f"📊 Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
    
    # Feature Selection (FS1) - fit on training data only
    if num_features is not None and feature_selection_method == 'FS1':
        if num_features > X_train.shape[1]:
            print(f"⚠️  Warning: Requested {num_features} features, but dataset only has {X_train.shape[1]}. Using all {X_train.shape[1]} features.")
            num_features = X_train.shape[1]
        
        if num_features < X_train.shape[1]:
            # Apply SelectKBest with f_classif - fit on training data only
            selector = SelectKBest(score_func=f_classif, k=num_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)  # Apply same transformation to test data
            
            # Get the selected feature indices for reporting
            selected_indices = selector.get_support(indices=True)
            print(f"🔍 Selected top {num_features} features: {selected_indices.tolist()}")
            
            X_train, X_test = X_train_selected, X_test_selected
    
    # Scale features to [0, 1] range - fit scaler on training data only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling to test data
    
    return X_train_scaled, X_test_scaled, y_train, y_test



# ============================================================================
# MODEL 1: VQC with ZZFeatureMap + EfficientSU2 (FS1 with specified features)
# ============================================================================
def run_model_1(sampler, estimator, num_features=4, num_train_samples=None):
    model1_accuracies = []
    train_features, test_features, train_labels, test_labels = load_and_prep_data(
        num_features=num_features, feature_selection_method='FS1', num_train_samples=num_train_samples
    )

    # Create and save circuit diagram on first iteration
    num_qubits = train_features.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=2)
    
    # Create the full circuit for visualization
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.compose(ansatz, inplace=True)
    
    # Save circuit diagram using text representation
    try:
        # Try to create a matplotlib circuit diagram
        fig = full_circuit.draw(output='mpl', style='iqp', plot_barriers=False)
        circuit_filename = f'plots/circuit_VQC_FS1_{num_features}features.png'
        fig.suptitle(f'VQC Circuit: ZZFeatureMap + EfficientSU2 ({num_features} features)', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(circuit_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Circuit diagram saved to {circuit_filename}")
    except Exception as e:
        # Fallback: save text representation
        circuit_text = str(full_circuit)
        circuit_filename = f'plots/circuit_VQC_FS1_{num_features}features.txt'
        with open(circuit_filename, 'w') as f:
            f.write(f"VQC Circuit: ZZFeatureMap + EfficientSU2 ({num_features} features)\n")
            f.write("=" * 60 + "\n")
            f.write(circuit_text)
        print(f"✅ Circuit text saved to {circuit_filename}")

    # Training loop with progress bar - train on same data, test on same test set
    for trial in tqdm(range(NUM_TRIALS), desc="Training Progress", ncols=80):
        classifier = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            loss="cross_entropy",
            optimizer=SLSQP(maxiter=MAX_ITER),
        )
        
        classifier.fit(train_features, train_labels)
        accuracy = classifier.score(test_features, test_labels)
        model1_accuracies.append(accuracy)

    return model1_accuracies, f'1. VQC ZZ+EfficientSU2 FS1-{num_features}F\n(HTRU_2 Binary)'

# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================
def analyze_and_visualize_results(accuracies, models):
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet', 'orange', 'pink']

    # Print statistics
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        mean_acc = np.mean(acc)
        std_acc = np.std(acc)
        min_acc = np.min(acc)
        max_acc = np.max(acc)
        print(f"{model.replace(chr(10), ' '):<30}: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes = axes.flatten()

    # Individual histograms
    for i, (ax, model, acc, color) in enumerate(zip(axes, models, accuracies, colors)):
        ax.hist(acc, bins=10, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(f'{model}\nMean: {np.mean(acc):.3f} ± {np.std(acc):.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    # Hide the unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f'IRIS QNN Model Comparison - {NUM_TRIALS} Trials Each', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('plots/iris_comparison_histograms2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Box plot comparison
    plt.figure(figsize=(14, 8))
    box_plot = plt.boxplot(accuracies, labels=[m.replace('\n', ' ') for m in models], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title(f'IRIS QNN Model Performance Comparison\n({NUM_TRIALS} Trials)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Model Architecture & Task', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')


    # Add mean markers
    means = [np.mean(acc) for acc in accuracies]
    plt.scatter(range(1, len(means) + 1), means, color='red', s=100, zorder=5, label='Mean')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/iris_comparison_boxplots2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical significance testing
    try:
        from scipy import stats
        import pandas as pd
        import seaborn as sns
        
        print("\n" + "=" * 60)
        print("📈 STATISTICAL ANALYSIS (Pairwise t-test p-values)")
        
        model_names = [m.split('\n')[0] for m in models]
        p_values = pd.DataFrame(np.ones((len(models), len(models))), index=model_names, columns=model_names)
        annotations = pd.DataFrame(np.full((len(models), len(models)), "", dtype=object), index=model_names, columns=model_names)
        
        for i in range(len(accuracies)):
            for j in range(i + 1, len(accuracies)):
                t_stat, p_value = stats.ttest_ind(accuracies[i], accuracies[j])
                p_values.iloc[i, j] = p_value
                p_values.iloc[j, i] = p_value # Symmetric matrix
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{model_names[i]:>18s} vs {model_names[j]:<18s}: p = {p_value:.4f} {significance}")
                
                annotations.iloc[i,j] = f"{p_value:.3f}\n{significance}"
                annotations.iloc[j,i] = f"{p_value:.3f}\n{significance}"


        # Create p-value heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(p_values, annot=annotations, fmt="s", cmap="coolwarm_r", linewidths=.5, vmin=0, vmax=0.1)
        plt.title('Pairwise t-test p-value Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/iris_comparison_p_values2.png', dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("\n📈 Statistical analysis requires scipy, pandas, and seaborn (pip install scipy pandas seaborn)")


def main():
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    print("🚀 HTRU_2 QNN Study")
    samples_text = f"{NUM_TRAIN_SAMPLES} train samples" if NUM_TRAIN_SAMPLES else "80/20 split"
    print(f"🔍 FS1 ({NUM_FEATURES} features) • {samples_text} • {NUM_TRIALS} trials")
    print("=" * 60)

    sampler = Sampler()
    estimator = Estimator()

    # Run Model 1 with the specified parameters
    accuracies, model_name = run_model_1(sampler, estimator, num_features=NUM_FEATURES, num_train_samples=NUM_TRAIN_SAMPLES)
    
    # Display results
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    print(f"\n✅ Results: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

    # Create enhanced accuracy distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram with enhanced statistics
    ax1.hist(accuracies, bins=15, alpha=0.75, color='skyblue', edgecolor='black', density=True)
    ax1.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    ax1.axvline(mean_acc + std_acc, color='orange', linestyle='dotted', linewidth=2, label=f'+1σ: {mean_acc + std_acc:.3f}')
    ax1.axvline(mean_acc - std_acc, color='orange', linestyle='dotted', linewidth=2, label=f'-1σ: {mean_acc - std_acc:.3f}')
    ax1.set_title(f'Accuracy Distribution - FS1 ({NUM_FEATURES} features)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Test Accuracy', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Box plot with statistics
    bp = ax2.boxplot([accuracies], labels=[f'FS1-{NUM_FEATURES}F'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax2.scatter([1], [mean_acc], color='red', s=100, zorder=5, label=f'Mean: {mean_acc:.3f}')
    ax2.set_title('Accuracy Summary Statistics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Add text box with detailed statistics
    stats_text = f"""Statistics Summary:
Mean: {mean_acc:.4f}
Std: {std_acc:.4f}
Min: {min_acc:.4f}
Max: {max_acc:.4f}
Trials: {NUM_TRIALS}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_filename = f'plots/gyro_fs1_{NUM_FEATURES}features_analysis.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Analysis saved to {plot_filename}")
    print(f"✨ Study completed!")


if __name__ == "__main__":
    main()