import numpy as np
from pprint import pprint


def generate_synthetic_vanishing_points_with_noise(ground_truth_points: np.array, scale: int = 2) -> np.array:
    """Generates synthetic vanishing points by adding random noise to the ground truth points.

    Parameters:
    ground_truth_points (np.array): Array of ground truth vanishing points with shape (num_points, 2).
    noise_range (float): Maximum value to add or subtract from each coordinate.
    scale (int): Scale of the shift of the synthetic vanishing point.

    Returns:
    np.array: Array of synthetic vanishing points with shape (num_points, 2)."""

    noisy_vanishing_points = ground_truth_points.copy()

    for index in range(ground_truth_points.shape[0]):
        x = ground_truth_points[index][0]
        y = ground_truth_points[index][1]

        noise_x = np.random.uniform(-x * scale, x * scale)
        noise_y = np.random.uniform(-y * scale, y * scale)

        noisy_vanishing_points[index][0] += noise_x
        noisy_vanishing_points[index][1] += noise_y

    pprint(f'the vanishing point: {ground_truth_points}')
    pprint(f'the synthetic point: {noisy_vanishing_points}')

    return noisy_vanishing_points
