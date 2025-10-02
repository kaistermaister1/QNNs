import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split

def create_line_boundary(x_position=0.5):
    """Create a vertical line boundary using Path."""
    # Create a vertical line from bottom to top of the unit square
    vertices = [
        [x_position, 0.0],    # Bottom of line
        [x_position, 1.0],    # Top of line
        [x_position, 1.0],    # Duplicate to close path
        [x_position, 0.0]     # Back to start
    ]
    
    return Path(vertices)

def generate_dataset(num_points=1000):
    """Generate random points with balanced left/right distribution."""
    np.random.seed(42)  # For reproducibility
    
    # Create line boundary
    line_path = create_line_boundary()
    
    # Calculate how many points we need on left and right
    num_left = num_points // 2
    num_right = num_points - num_left  # Handle odd numbers
    
    # Generate many random points to have enough on left and right
    num_candidates = max(2000, num_points * 10)  # Scale with desired points
    candidate_points = np.random.uniform(0, 1, (num_candidates, 2))
    
    # Check which points are on the left of the line (x < 0.5)
    left_mask = candidate_points[:, 0] < 0.5
    
    # Get points on left and right
    left_candidates = candidate_points[left_mask]
    right_candidates = candidate_points[~left_mask]
    
    # Select the required number from each group
    left_points = left_candidates[:num_left] if len(left_candidates) >= num_left else left_candidates
    right_points = right_candidates[:num_right] if len(right_candidates) >= num_right else right_candidates
    
    # If we don't have enough points in either category, generate more
    while len(left_points) < num_left or len(right_points) < num_right:
        additional_points = np.random.uniform(0, 1, (1000, 2))
        additional_left_mask = additional_points[:, 0] < 0.5
        
        if len(left_points) < num_left:
            additional_left = additional_points[additional_left_mask]
            left_points = np.vstack([left_points, additional_left])[:num_left]
        
        if len(right_points) < num_right:
            additional_right = additional_points[~additional_left_mask]
            right_points = np.vstack([right_points, additional_right])[:num_right]
    
    return left_points, right_points, line_path

def create_labeled_dataset(num_points=100):
    """Create labeled dataset with scalar labels (0 or 1) and 80/20 split."""
    left_points, right_points, line_path = generate_dataset(num_points)
    
    # Combine all points
    all_points = np.vstack([left_points, right_points])
    
    # Create scalar labels: left -> 0, right -> 1
    labels = np.hstack([np.zeros(len(left_points)), np.ones(len(right_points))])
    
    # Split into train/test while maintaining proportions
    X_train, X_test, y_train, y_test = train_test_split(
        all_points, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, line_path

def visualize_data(num_points=100):
    """Visualize the line boundary dataset with train/test split."""
    X_train, X_test, y_train, y_test, line_path = get_line_data(num_points)
    
    # Labels are already scalar: 0 for left, 1 for right
    y_train_binary = y_train.astype(int)
    y_test_binary = y_test.astype(int)
    
    # Separate train points by class  
    train_left = X_train[y_train_binary == 0]  # Class 0 is left
    train_right = X_train[y_train_binary == 1]  # Class 1 is right
    test_left = X_test[y_test_binary == 0]
    test_right = X_test[y_test_binary == 1]
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for train and test
    plt.subplot(1, 2, 1)
    
    # Plot training points
    plt.scatter(train_left[:, 0], train_left[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Left Train ({len(train_left)} points)', marker='o')
    plt.scatter(train_right[:, 0], train_right[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Right Train ({len(train_right)} points)', marker='o')
    
    # Draw line boundary
    plt.axvline(x=0.5, color='black', linestyle=':', linewidth=3, label='Line boundary')
    
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
    
    plt.scatter(test_left[:, 0], test_left[:, 1], 
               c='red', s=50, alpha=0.7, label=f'Left Test ({len(test_left)} points)', marker='s')
    plt.scatter(test_right[:, 0], test_right[:, 1], 
               c='blue', s=50, alpha=0.7, label=f'Right Test ({len(test_right)} points)', marker='s')
    
    plt.axvline(x=0.5, color='black', linestyle=':', linewidth=3, label='Line boundary')
    
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
    print(f"  - Left (red): {len(train_left)} points")
    print(f"  - Right (blue): {len(train_right)} points")
    print(f"  - Total train: {len(X_train)} points")
    print(f"\nTest Set (20%):")
    print(f"  - Left (red): {len(test_left)} points")
    print(f"  - Right (blue): {len(test_right)} points")
    print(f"  - Total test: {len(X_test)} points")
    
    print(f"\nLabel formats:")
    print(f"Scalar labels:")
    print(f"Left: 0, Right: 1")

def get_line_data(num_points=100):
    """
    Get all line classification data without plotting.
    
    Args:
        num_points: Total number of data points to generate (default: 100)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, line_path)
            - X_train: Training features (coordinates)
            - X_test: Test features (coordinates) 
            - y_train: Training scalar labels (0 or 1)
            - y_test: Test scalar labels (0 or 1)
            - line_path: The line boundary path object
            
    Label encoding:
        - Left of line: 0
        - Right of line: 1
    """
    return create_labeled_dataset(num_points)

if __name__ == "__main__":
    # Demonstrate with different dataset sizes
    print("Demonstrating configurable dataset sizes:")
    print("=" * 50)
    
    # Test with different sizes
    sizes_to_test = [50, 100, 200, 500]
    
    for size in sizes_to_test:
        print(f"\nTesting with {size} total points:")
        X_train, X_test, y_train, y_test, line_path = get_line_data(size)
        
        # Count left/right for each split
        train_left_count = np.sum(y_train == 0)
        train_right_count = np.sum(y_train == 1)
        test_left_count = np.sum(y_test == 0)
        test_right_count = np.sum(y_test == 1)
        
        print(f"  Training: {train_left_count} left, {train_right_count} right")
        print(f"  Test: {test_left_count} left, {test_right_count} right")
        print(f"  Total: {len(X_train) + len(X_test)} points")
    
    print("\n" + "=" * 50)
    print("Showing visualization with 200 points:")
    
    # Show visualization with default size
    visualize_data(200)
    
    # Get data for demonstration
    X_train, X_test, y_train, y_test, line_path = get_line_data(200)
    
    # Demonstrate the data formats
    print(f"\nExample data points:")
    print(f"First training point: {X_train[0]} -> Scalar label: {y_train[0]}")
    print(f"First test point: {X_test[0]} -> Scalar label: {y_test[0]}")
    
    # Show shapes
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Show label meaning
    print(f"\nLabel encoding:")
    print(f"Left of line: 0")
    print(f"Right of line: 1")
    
    print(f"\nTo use custom dataset sizes in your code:")
    print(f"train_features, test_features, train_labels, test_labels, boundary = get_line_data(500)") 