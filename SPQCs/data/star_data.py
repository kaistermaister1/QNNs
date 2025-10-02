import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from sklearn.model_selection import train_test_split

def create_star_boundary(center=(0.5, 0.5), outer_radius=0.45, inner_radius=0.2, num_points=5):
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

def generate_dataset(num_points=2000, condense=False):
    """Generate random points with balanced inside/outside distribution."""
    np.random.seed(42)  # For reproducibility
    
    # Create star boundary
    star_path = create_star_boundary()
    
    # Calculate how many points we need inside and outside
    num_inside = num_points // 2
    num_outside = num_points - num_inside  # Handle odd numbers
    
    if condense:
        # Generate condensed outside points: mix of concentrated near boundary and random distribution
        vertices = star_path.vertices[:-1]  # Remove duplicate last vertex
        center = np.mean(vertices, axis=0)
        
        # Generate inside points normally
        inside_points = []
        attempts = 0
        while len(inside_points) < num_inside and attempts < num_inside * 50:
            candidate = np.random.uniform(0, 1, 2)
            if star_path.contains_points(np.array([candidate]))[0]:
                inside_points.append(candidate)
            attempts += 1
        inside_points = np.array(inside_points)
        
        # Split outside points: 70% concentrated near boundary, 40% randomly distributed
        num_concentrated = int(num_outside * 0.7)
        num_random = num_outside - num_concentrated
        
        # Generate concentrated points near boundary
        concentrated_points = []
        attempts = 0
        max_attempts = num_concentrated * 200
        band_thickness = 0.05  # Fixed thickness of the band around star
        
        while len(concentrated_points) < num_concentrated and attempts < max_attempts:
            # Generate random angle around the star
            angle = np.random.uniform(0, 2*np.pi)
            
            # Find the distance from center to star boundary at this angle
            # Sample multiple points along this ray to find boundary
            ray_length = 0.5  # Maximum ray length
            ray_points = []
            for r in np.linspace(0, ray_length, 100):
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                ray_points.append([x, y])
            
            ray_points = np.array(ray_points)
            inside_mask = star_path.contains_points(ray_points)
            
            # Find the boundary point (last inside point)
            if np.any(inside_mask):
                last_inside_idx = np.where(inside_mask)[0][-1]
                boundary_distance = np.linalg.norm(ray_points[last_inside_idx] - center)
            else:
                # If no inside points found, use minimum distance
                boundary_distance = 0.1
            
            # Generate point in band outside boundary
            band_start = boundary_distance + 0.01  # Small gap from boundary
            band_end = boundary_distance + band_thickness
            
            r = np.random.uniform(band_start, band_end)
            candidate = center + r * np.array([np.cos(angle), np.sin(angle)])
            
            # Add small random jitter perpendicular to boundary
            perp_angle = angle + np.pi/2
            jitter_magnitude = np.random.uniform(0, 0.01)
            jitter = jitter_magnitude * np.array([np.cos(perp_angle), np.sin(perp_angle)])
            candidate = candidate + jitter
            
            # Check bounds and star containment
            if (0 <= candidate[0] <= 1 and 0 <= candidate[1] <= 1 and 
                not star_path.contains_points(np.array([candidate]))[0]):
                concentrated_points.append(candidate)
            
            attempts += 1
        
        # Generate random points outside the star
        random_points = []
        attempts = 0
        max_attempts = num_random * 50
        
        while len(random_points) < num_random and attempts < max_attempts:
            candidate = np.random.uniform(0, 1, 2)
            if not star_path.contains_points(np.array([candidate]))[0]:
                random_points.append(candidate)
            attempts += 1
        
        # Combine concentrated and random points
        concentrated_points = np.array(concentrated_points) if concentrated_points else np.empty((0, 2))
        random_points = np.array(random_points) if random_points else np.empty((0, 2))
        outside_points = np.vstack([concentrated_points, random_points]) if len(concentrated_points) > 0 and len(random_points) > 0 else (
            concentrated_points if len(concentrated_points) > 0 else random_points
        )
        
    else:
        # Standard generation
        num_candidates = max(2000, num_points * 10)
        candidate_points = np.random.uniform(0, 1, (num_candidates, 2))
        
        # Check which points are inside the star
        inside_mask = star_path.contains_points(candidate_points)
        
        # Get points inside and outside
        inside_candidates = candidate_points[inside_mask]
        outside_candidates = candidate_points[~inside_mask]
        
        # Select the required number from each group
        inside_points = inside_candidates[:num_inside] if len(inside_candidates) >= num_inside else inside_candidates
        outside_points = outside_candidates[:num_outside] if len(outside_candidates) >= num_outside else outside_candidates
        
        # If we don't have enough points in either category, generate more
        while len(inside_points) < num_inside or len(outside_points) < num_outside:
            additional_points = np.random.uniform(0, 1, (1000, 2))
            additional_inside_mask = star_path.contains_points(additional_points)
            
            if len(inside_points) < num_inside:
                additional_inside = additional_points[additional_inside_mask]
                inside_points = np.vstack([inside_points, additional_inside])[:num_inside]
            
            if len(outside_points) < num_outside:
                additional_outside = additional_points[~additional_inside_mask]
                outside_points = np.vstack([outside_points, additional_outside])[:num_outside]

    return inside_points, outside_points, star_path



def create_labeled_dataset(num_points=100, condense=False):
    """Create labeled dataset with scalar labels (0 or 1) and 80/20 split."""
    inside_points, outside_points, star_path = generate_dataset(num_points, condense)
    
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
    
    return X_train, X_test, y_train, y_test, star_path

def visualize_data(num_points=100, data=None):
    """Visualize the star boundary dataset with train/test split."""
    if data is not None:
        X_train, X_test, y_train, y_test, star_path = data
    else:
        X_train, X_test, y_train, y_test, star_path = get_star_data(num_points)
    
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
    print(f"Scalar labels:")
    print(f"Inside: 0, Outside: 1")

def get_star_data(num_points=100, condense=False):
    """
    Get all star classification data without plotting.
    
    Args:
        num_points: Total number of data points to generate (default: 100)
        condense: If True, concentrate outside points near star boundary (default: False)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, star_path)
            - X_train: Training features (coordinates)
            - X_test: Test features (coordinates) 
            - y_train: Training scalar labels (0 or 1)
            - y_test: Test scalar labels (0 or 1)
            - star_path: The star boundary path object
            
    Label encoding:
        - Inside star: 0
        - Outside star: 1
    """
    return create_labeled_dataset(num_points, condense)

if __name__ == "__main__":
    # Generate and visualize condensed star dataset
    condensed_data = get_star_data(2000, condense=True)
    X_train, X_test, y_train, y_test, star_path = condensed_data
    print(f"Generated: {len(X_train)} train, {len(X_test)} test points (condensed outside points near boundary)")
    
    # Visualize the condensed dataset
    visualize_data(data=condensed_data)