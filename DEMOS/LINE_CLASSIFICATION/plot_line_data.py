#!/usr/bin/env python3
"""
Simple script to visualize line classification data with y = -x boundary
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit_algorithms.utils import algorithm_globals

def generate_line_dataset(num_samples=200, seed=42):
    """Generate line classification dataset with y = -x boundary."""
    algorithm_globals.random_seed = seed
    
    # Generate random points in [-1, 1] x [-1, 1]
    X = 2 * algorithm_globals.random.random([num_samples, 2]) - 1
    
    # Classification: points above/below y = -x line
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # x + y >= 0 means above y = -x
    y = 2 * y01 - 1  # Convert to {-1, +1} labels
    
    return X, y

def plot_line_data(num_points=200):
    """Plot line classification data with y = -x boundary line."""
    
    # Generate the data
    X, y = generate_line_dataset(num_points)
    
    # Separate points by class
    below_line = X[y == -1]  # Class -1: below y = -x line
    above_line = X[y == 1]   # Class +1: above y = -x line
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot points by class
    plt.scatter(below_line[:, 0], below_line[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Class -1 (Below): {len(below_line)} points')
    plt.scatter(above_line[:, 0], above_line[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Class +1 (Above): {len(above_line)} points')
    
    # Draw the dotted boundary line y = -x
    x_line = np.linspace(-1, 1, 100)
    y_line = -x_line
    plt.plot(x_line, y_line, 'k:', linewidth=2, label='Decision boundary: y = -x')
    
    # Format the plot
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'Line Classification Data ({num_points} total points)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Data summary:")
    print(f"  Total points: {len(X)}")
    print(f"  Class -1 (below y = -x): {len(below_line)} points")
    print(f"  Class +1 (above y = -x): {len(above_line)} points")
    print(f"  Decision boundary: y = -x (dotted line)")

if __name__ == "__main__":
    plot_line_data()