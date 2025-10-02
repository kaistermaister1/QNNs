import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split

def create_circle_boundary(center=(0.5, 0.5), radius=0.3):
    """Create a circular boundary using Path."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    vertices = np.column_stack([x, y])
    return Path(vertices)

def generate_dataset(num_points=2000):
    """Generate random points with balanced inside/outside distribution."""
    np.random.seed(42)  # For reproducibility
    
    # Create circle boundary
    circle_path = create_circle_boundary()
    
    # Calculate how many points we need inside and outside
    num_inside = num_points // 2
    num_outside = num_points - num_inside  # Handle odd numbers
    
    # Standard generation
    num_candidates = max(2000, num_points * 10)
    candidate_points = np.random.uniform(0, 1, (num_candidates, 2))
    
    # Check which points are inside the circle
    inside_mask = circle_path.contains_points(candidate_points)
    
    # Get points inside and outside
    inside_candidates = candidate_points[inside_mask]
    outside_candidates = candidate_points[~inside_mask]
    
    # Select the required number from each group
    inside_points = inside_candidates[:num_inside] if len(inside_candidates) >= num_inside else inside_candidates
    outside_points = outside_candidates[:num_outside] if len(outside_candidates) >= num_outside else outside_candidates
    
    # If we don't have enough points in either category, generate more
    while len(inside_points) < num_inside or len(outside_points) < num_outside:
        additional_points = np.random.uniform(0, 1, (1000, 2))
        additional_inside_mask = circle_path.contains_points(additional_points)
        
        if len(inside_points) < num_inside:
            additional_inside = additional_points[additional_inside_mask]
            inside_points = np.vstack([inside_points, additional_inside])[:num_inside]
        
        if len(outside_points) < num_outside:
            additional_outside = additional_points[~additional_inside_mask]
            outside_points = np.vstack([outside_points, additional_outside])[:num_outside]

    return inside_points, outside_points, circle_path


def create_labeled_dataset(num_points=100):
    """Create labeled dataset with scalar labels (0 or 1) and 80/20 split."""
    inside_points, outside_points, circle_path = generate_dataset(num_points)
    
    # Combine all points
    all_points = np.vstack([inside_points, outside_points])
    
    # Create scalar labels: inside -> 0, outside -> 1
    labels = np.hstack([np.zeros(len(inside_points)), np.ones(len(outside_points))])
    
    # Split into train/test while maintaining proportions
    X_train, X_test, y_train, y_test = train_test_split(
        all_points, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, circle_path

def visualize_data(num_points=100, data=None):
    """Visualize the circle boundary dataset with train/test split."""
    if data is not None:
        X_train, X_test, y_train, y_test, circle_path = data
    else:
        X_train, X_test, y_train, y_test, circle_path = get_circle_data(num_points)
    
    # Labels are already scalar: 0 for inside, 1 for outside
    y_train_binary = y_train.astype(int)
    y_test_binary = y_test.astype(int)
    
    # Separate train points by class  
    train_inside = X_train[y_train_binary == 0]  # Class 0 is inside
    train_outside = X_train[y_train_binary == 1]  # Class 1 is outside
    test_inside = X_test[y_test_binary == 0]
    test_outside = X_test[y_test_binary == 1]
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for train and test
    plt.subplot(1, 2, 1)
    
    # Plot training points
    plt.scatter(train_inside[:, 0], train_inside[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Inside Train ({len(train_inside)} points)', marker='o')
    plt.scatter(train_outside[:, 0], train_outside[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Outside Train ({len(train_outside)} points)', marker='o')
    
    # Draw circle boundary
    vertices = circle_path.vertices
    circle_x = vertices[:, 0]
    circle_y = vertices[:, 1]
    plt.plot(circle_x, circle_y, 'k:', linewidth=3, label='Circle boundary')
    
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
    
    plt.plot(circle_x, circle_y, 'k:', linewidth=3, label='Circle boundary')
    
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
    print(f"Scalar labels:")
    print(f"Inside: 0, Outside: 1")

def get_circle_data(num_points=100):
    """
    Get all circle classification data without plotting.
    
    Args:
        num_points: Total number of data points to generate (default: 100)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, circle_path)
            - X_train: Training features (coordinates)
            - X_test: Test features (coordinates) 
            - y_train: Training scalar labels (0 or 1)
            - y_test: Test scalar labels (0 or 1)
            - circle_path: The circle boundary path object
            
    Label encoding:
        - Inside circle: 0
        - Outside circle: 1
    """
    return create_labeled_dataset(num_points)

if __name__ == "__main__":
    # Generate and visualize circle dataset
    print("Demonstrating circle dataset:")
    print("=" * 50)
    
    # Test different sizes
    sizes_to_test = [50, 100, 200, 500]
    for size in sizes_to_test:
        print(f"\nTesting with {size} total points:")
        X_train, X_test, y_train, y_test, circle_path = get_circle_data(size)
        train_inside_count = np.sum(y_train == 0)
        train_outside_count = np.sum(y_train == 1)
        test_inside_count = np.sum(y_test == 0)
        test_outside_count = np.sum(y_test == 1)
        print(f"  Training: {train_inside_count} inside, {train_outside_count} outside")
        print(f"  Test: {test_inside_count} inside, {test_outside_count} outside")
        print(f"  Total: {len(X_train) + len(X_test)} points")
    
    print("\n" + "=" * 50)
    print("Showing visualization with 1000 points:")
    data = get_circle_data(1000)
    visualize_data(data=data)
    
    X_train, X_test, y_train, y_test, circle_path = data
    
    print(f"\nExample data points:")
    print(f"First training point: {X_train[0]} -> Scalar label: {y_train[0]}")
    print(f"First test point: {X_test[0]} -> Scalar label: {y_test[0]}")
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"\nLabel encoding:")
    print(f"Inside circle: 0")
    print(f"Outside circle: 1")
    print(f"\nTo use custom dataset sizes in your code:")
    print(f"train_features, test_features, train_labels, test_labels, circle_path = get_circle_data(500)")