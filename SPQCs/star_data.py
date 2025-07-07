import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split

def create_star_boundary(center=(0.5, 0.5), outer_radius=0.3, inner_radius=0.15, num_points=5):
    """Create a star-shaped boundary using Path."""
    angles = np.linspace(0, 2*np.pi, num_points*2, endpoint=False)
    
    # Alternate between outer and inner radius points
    vertices = []
    for i, angle in enumerate(angles):
        if i % 2 == 0:  # Outer points
            radius = outer_radius
        else:  # Inner points
            radius = inner_radius
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append([x, y])
    
    # Close the path
    vertices.append(vertices[0])
    
    return Path(vertices)

def generate_dataset():
    """Generate 100 random points with exactly 50 inside and 50 outside the star boundary."""
    np.random.seed(42)  # For reproducibility
    
    # Create star boundary
    star_path = create_star_boundary()
    
    # Generate many random points to have enough inside and outside
    num_candidates = 2000
    candidate_points = np.random.uniform(0, 1, (num_candidates, 2))
    
    # Check which points are inside the star
    inside_mask = star_path.contains_points(candidate_points)
    
    # Get points inside and outside
    inside_candidates = candidate_points[inside_mask]
    outside_candidates = candidate_points[~inside_mask]
    
    # Select exactly 50 from each group
    inside_points = inside_candidates[:50] if len(inside_candidates) >= 50 else inside_candidates
    outside_points = outside_candidates[:50] if len(outside_candidates) >= 50 else outside_candidates
    
    # If we don't have enough points in either category, generate more
    while len(inside_points) < 50 or len(outside_points) < 50:
        additional_points = np.random.uniform(0, 1, (1000, 2))
        additional_inside_mask = star_path.contains_points(additional_points)
        
        if len(inside_points) < 50:
            additional_inside = additional_points[additional_inside_mask]
            inside_points = np.vstack([inside_points, additional_inside])[:50]
        
        if len(outside_points) < 50:
            additional_outside = additional_points[~additional_inside_mask]
            outside_points = np.vstack([outside_points, additional_outside])[:50]
    
    return inside_points, outside_points, star_path

def create_labeled_dataset():
    """Create labeled dataset with one-hot encoding and 80/20 split."""
    inside_points, outside_points, star_path = generate_dataset()
    
    # Combine all points
    all_points = np.vstack([inside_points, outside_points])
    
    # Create labels (1 for inside, 0 for outside)
    labels = np.hstack([np.ones(len(inside_points)), np.zeros(len(outside_points))])
    
    # Create one-hot labels
    one_hot_labels = np.zeros((len(labels), 2))
    one_hot_labels[labels == 1, 0] = 1  # Inside: [1, 0]
    one_hot_labels[labels == 0, 1] = 1  # Outside: [0, 1]
    
    # Split into train/test while maintaining proportions
    # We'll use stratify to ensure both classes are represented proportionally
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
        all_points, labels, one_hot_labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, star_path

def visualize_data():
    """Visualize the star boundary dataset with train/test split."""
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, star_path = get_star_data()
    
    # Separate train points by class
    train_inside = X_train[y_train == 1]
    train_outside = X_train[y_train == 0]
    test_inside = X_test[y_test == 1]
    test_outside = X_test[y_test == 0]
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for train and test
    plt.subplot(1, 2, 1)
    
    # Plot training points
    plt.scatter(train_inside[:, 0], train_inside[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Inside Train ({len(train_inside)} points)', marker='o')
    plt.scatter(train_outside[:, 0], train_outside[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Outside Train ({len(train_outside)} points)', marker='o')
    
    # Draw star boundary
    vertices = star_path.vertices
    star_x = vertices[:, 0]
    star_y = vertices[:, 1]
    plt.plot(star_x, star_y, 'k:', linewidth=3, label='Star boundary')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Training Set (80%)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Plot test set
    plt.subplot(1, 2, 2)
    
    plt.scatter(test_inside[:, 0], test_inside[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Inside Test ({len(test_inside)} points)', marker='s')
    plt.scatter(test_outside[:, 0], test_outside[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Outside Test ({len(test_outside)} points)', marker='s')
    
    plt.plot(star_x, star_y, 'k:', linewidth=3, label='Star boundary')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Test Set (20%)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total points: {len(X_train) + len(X_test)}")
    print(f"\nTraining Set (80%):")
    print(f"  - Inside (red): {len(train_inside)} points")
    print(f"  - Outside (blue): {len(train_outside)} points")
    print(f"  - Total train: {len(X_train)} points")
    print(f"\nTest Set (20%):")
    print(f"  - Inside (red): {len(test_inside)} points")
    print(f"  - Outside (blue): {len(test_outside)} points")
    print(f"  - Total test: {len(X_test)} points")
    
    print(f"\nLabel formats:")
    print(f"Binary labels - Inside: 1, Outside: 0")
    print(f"One-hot labels - Inside: [1, 0], Outside: [0, 1]")

def get_star_data():
    """
    Get all star classification data without plotting.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, star_path)
            - X_train: Training features (coordinates)
            - X_test: Test features (coordinates) 
            - y_train: Training binary labels (1=inside, 0=outside)
            - y_test: Test binary labels
            - y_train_onehot: Training one-hot labels ([1,0]=inside, [0,1]=outside)
            - y_test_onehot: Test one-hot labels
            - star_path: The star boundary path object
    """
    return create_labeled_dataset()

if __name__ == "__main__":
    # Show visualization and demo when run directly
    visualize_data()
    
    # Get data for demonstration
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, star_path = get_star_data()
    
    # Demonstrate the data formats
    print(f"\nExample data points:")
    print(f"First training point: {X_train[0]} -> Binary label: {y_train[0]} -> One-hot: {y_train_onehot[0]}")
    print(f"First test point: {X_test[0]} -> Binary label: {y_test[0]} -> One-hot: {y_test_onehot[0]}")
    
    # Show shapes
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, y_train_onehot: {y_train_onehot.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, y_test_onehot: {y_test_onehot.shape}")