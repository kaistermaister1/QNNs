import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split


def create_blobs_boundaries(centers, radii):
    """Create a list of circular blob boundaries as Path objects."""
    boundaries = []
    for center, radius in zip(centers, radii):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        vertices = np.column_stack([x, y])
        boundaries.append(Path(vertices))
    return boundaries

def generate_blobs_dataset(num_points=1000, centers=None, radii=None):
    """Generate random points with balanced inside/outside distribution for 3 blobs."""
    np.random.seed(42)  # For reproducibility
    if centers is None:
        centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    if radii is None:
        radii = [0.15, 0.12, 0.13]
    boundaries = create_blobs_boundaries(centers, radii)

    # Calculate how many points we need inside and outside
    num_inside = num_points // 2
    num_outside = num_points - num_inside

    # Generate many random points to have enough inside and outside
    num_candidates = max(2000, num_points * 10)
    candidate_points = np.random.uniform(0, 1, (num_candidates, 2))

    # Check which points are inside any blob
    inside_mask = np.zeros(num_candidates, dtype=bool)
    for path in boundaries:
        inside_mask |= path.contains_points(candidate_points)

    inside_candidates = candidate_points[inside_mask]
    outside_candidates = candidate_points[~inside_mask]

    # Select the required number from each group
    inside_points = inside_candidates[:num_inside] if len(inside_candidates) >= num_inside else inside_candidates
    outside_points = outside_candidates[:num_outside] if len(outside_candidates) >= num_outside else outside_candidates

    # If we don't have enough points in either category, generate more
    while len(inside_points) < num_inside or len(outside_points) < num_outside:
        additional_points = np.random.uniform(0, 1, (1000, 2))
        additional_inside_mask = np.zeros(1000, dtype=bool)
        for path in boundaries:
            additional_inside_mask |= path.contains_points(additional_points)
        if len(inside_points) < num_inside:
            additional_inside = additional_points[additional_inside_mask]
            inside_points = np.vstack([inside_points, additional_inside])[:num_inside]
        if len(outside_points) < num_outside:
            additional_outside = additional_points[~additional_inside_mask]
            outside_points = np.vstack([outside_points, additional_outside])[:num_outside]

    return inside_points, outside_points, boundaries

def create_labeled_dataset(num_points=100):
    """Create labeled dataset with scalar labels (0 or 1) and 80/20 split."""
    inside_points, outside_points, boundaries = generate_blobs_dataset(num_points)
    all_points = np.vstack([inside_points, outside_points])
    labels = np.hstack([np.zeros(len(inside_points)), np.ones(len(outside_points))])
    X_train, X_test, y_train, y_test = train_test_split(
        all_points, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return X_train, X_test, y_train, y_test, boundaries

def visualize_data(num_points=100):
    """Visualize the blobs dataset with train/test split."""
    X_train, X_test, y_train, y_test, boundaries = get_blobs_data(num_points)
    y_train_binary = y_train.astype(int)
    y_test_binary = y_test.astype(int)
    train_inside = X_train[y_train_binary == 0]
    train_outside = X_train[y_train_binary == 1]
    test_inside = X_test[y_test_binary == 0]
    test_outside = X_test[y_test_binary == 1]
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(train_inside[:, 0], train_inside[:, 1], c='red', s=50, alpha=0.7, label=f'Inside Train ({len(train_inside)} points)', marker='o')
    plt.scatter(train_outside[:, 0], train_outside[:, 1], c='blue', s=50, alpha=0.7, label=f'Outside Train ({len(train_outside)} points)', marker='o')
    for path in boundaries:
        vertices = path.vertices
        plt.plot(vertices[:, 0], vertices[:, 1], 'k:', linewidth=2, label='Blob boundary')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Training Set (80%)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplot(1, 2, 2)
    plt.scatter(test_inside[:, 0], test_inside[:, 1], c='red', s=50, alpha=0.7, label=f'Inside Test ({len(test_inside)} points)', marker='s')
    plt.scatter(test_outside[:, 0], test_outside[:, 1], c='blue', s=50, alpha=0.7, label=f'Outside Test ({len(test_outside)} points)', marker='s')
    for path in boundaries:
        vertices = path.vertices
        plt.plot(vertices[:, 0], vertices[:, 1], 'k:', linewidth=2, label='Blob boundary')
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

def get_blobs_data(num_points=100):
    """
    Get all blobs classification data without plotting.
    Args:
        num_points: Total number of data points to generate (default: 100)
    Returns:
        tuple: (X_train, X_test, y_train, y_test, boundaries)
            - X_train: Training features (coordinates)
            - X_test: Test features (coordinates)
            - y_train: Training scalar labels (0 or 1)
            - y_test: Training scalar labels (0 or 1)
            - boundaries: List of Path objects for the blob boundaries
    Label encoding:
        - Inside any blob: 0
        - Outside all blobs: 1
    """
    return create_labeled_dataset(num_points)

if __name__ == "__main__":
    print("Demonstrating blobs dataset:")
    print("=" * 50)
    sizes_to_test = [50, 100, 200, 500]
    for size in sizes_to_test:
        print(f"\nTesting with {size} total points:")
        X_train, X_test, y_train, y_test, boundaries = get_blobs_data(size)
        train_inside_count = np.sum(y_train == 0)
        train_outside_count = np.sum(y_train == 1)
        test_inside_count = np.sum(y_test == 0)
        test_outside_count = np.sum(y_test == 1)
        print(f"  Training: {train_inside_count} inside, {train_outside_count} outside")
        print(f"  Test: {test_inside_count} inside, {test_outside_count} outside")
        print(f"  Total: {len(X_train) + len(X_test)} points")
    print("\n" + "=" * 50)
    print("Showing visualization with 200 points:")
    visualize_data(200)
    X_train, X_test, y_train, y_test, boundaries = get_blobs_data(200)
    print(f"\nExample data points:")
    print(f"First training point: {X_train[0]} -> Scalar label: {y_train[0]}")
    print(f"First test point: {X_test[0]} -> Scalar label: {y_test[0]}")
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"\nLabel encoding:")
    print(f"Inside any blob: 0")
    print(f"Outside all blobs: 1")
    print(f"\nTo use custom dataset sizes in your code:")
    print(f"train_features, test_features, train_labels, test_labels, boundaries = get_blobs_data(500)") 