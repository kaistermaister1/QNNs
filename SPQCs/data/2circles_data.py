import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split

def create_2circles_boundaries(centers, radii):
    """Create a list of circular blob boundaries as Path objects (2 blobs)."""
    boundaries = []
    for center, radius in zip(centers, radii):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        vertices = np.column_stack([x, y])
        boundaries.append(Path(vertices))
    return boundaries

def generate_2circles_dataset(num_points=1000, centers=None, radii=None):
    """Generate random points with balanced inside/outside distribution for 2 blobs, and at least 80 between the blobs."""
    np.random.seed(42)  # For reproducibility
    if centers is None:
        centers = [(0.35, 0.35), (0.65, 0.65)]
    if radii is None:
        radii = [0.18, 0.18]
    boundaries = create_2circles_boundaries(centers, radii)

    num_inside = num_points // 2
    num_outside = num_points - num_inside
    num_between = 80
    num_candidates = max(2000, num_points * 10)

    def is_between(point):
        # Not in either blob
        in_blob = any(path.contains_point(point) for path in boundaries)
        if in_blob:
            return False
        # Band ("sausage") between centers
        c0 = np.array(centers[0])
        c1 = np.array(centers[1])
        p = np.array(point)
        v = c1 - c0
        v_norm = v / np.linalg.norm(v)
        proj = np.dot(p - c0, v_norm)
        # Only between the centers
        if proj < 0 or proj > np.linalg.norm(v):
            return False
        # Distance from the line
        dist_to_line = np.linalg.norm((p - c0) - proj * v_norm)
        # Accept points within a certain width (e.g., 0.1)
        return dist_to_line < 0.1

    # Generate many random points to have enough inside, outside, and between
    while True:
        candidate_points = np.random.uniform(0, 1, (num_candidates, 2))
        inside_mask = np.zeros(num_candidates, dtype=bool)
        for path in boundaries:
            inside_mask |= path.contains_points(candidate_points)
        inside_candidates = candidate_points[inside_mask]
        outside_candidates = candidate_points[~inside_mask]
        between_mask = np.array([is_between(pt) for pt in outside_candidates])
        between_candidates = outside_candidates[between_mask]
        if len(inside_candidates) >= num_inside and len(outside_candidates) >= num_outside and len(between_candidates) >= num_between:
            break
        num_candidates += 1000  # Try more points if not enough

    inside_points = inside_candidates[:num_inside]
    # For outside, ensure at least 20 between points are included
    between_points = between_candidates[:num_between]
    # Fill the rest of outside points from the remaining outside_candidates (excluding those already in between_points)
    mask_not_between = np.ones(len(outside_candidates), dtype=bool)
    mask_not_between[np.where(between_mask)[0][:num_between]] = False
    outside_rest = outside_candidates[mask_not_between][:num_outside - num_between]
    outside_points = np.vstack([between_points, outside_rest])

    return inside_points, outside_points, boundaries

def create_labeled_dataset(num_points=100):
    inside_points, outside_points, boundaries = generate_2circles_dataset(num_points)
    all_points = np.vstack([inside_points, outside_points])
    labels = np.hstack([np.zeros(len(inside_points)), np.ones(len(outside_points))])
    X_train, X_test, y_train, y_test = train_test_split(
        all_points, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return X_train, X_test, y_train, y_test, boundaries

def visualize_data(num_points=100):
    X_train, X_test, y_train, y_test, boundaries = get_2circles_data(num_points)
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

def get_2circles_data(num_points=100):
    """
    Get all 2circles classification data without plotting.
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
    print("Demonstrating 2circles dataset:")
    print("=" * 50)
    sizes_to_test = [50, 100, 200, 500]
    for size in sizes_to_test:
        print(f"\nTesting with {size} total points:")
        X_train, X_test, y_train, y_test, boundaries = get_2circles_data(size)
        train_inside_count = np.sum(y_train == 0)
        train_outside_count = np.sum(y_train == 1)
        test_inside_count = np.sum(y_test == 0)
        test_outside_count = np.sum(y_test == 1)
        print(f"  Training: {train_inside_count} inside, {train_outside_count} outside")
        print(f"  Test: {test_inside_count} inside, {test_outside_count} outside")
        print(f"  Total: {len(X_train) + len(X_test)} points")
    print("\n" + "=" * 50)
    print("Showing visualization with 200 points:")
    visualize_data(1000)
    X_train, X_test, y_train, y_test, boundaries = get_2circles_data(200)
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
    print(f"train_features, test_features, train_labels, test_labels, boundaries = get_2circles_data(500)") 