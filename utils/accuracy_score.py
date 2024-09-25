import numpy as np


def accuracy_score(vanishing_point: np.array,
                   predicted_point: np.array,
                   max_dist: float = 5.0) -> float:
    """This function returns the overall accuracy for the vanishing point detection.

    Parameters:
    vanishing_point (np.array): The true vanishing point coordinates.
    predicted_point (np.array): The predicted vanishing point coordinates.
    max_dist (float): The threshold distance. If bigger replacing with that value.

    Returns:
    The overall accuracy of the vanishing point detection.
    """

    weights = []  # Weights for computing the final overall weighted accuracy.
    accs = []  # List of accuracies for each vanishing point.
    from_center_distances = []  # Distance of the point from the center.

    for index in range(len(vanishing_point)):
        if vanishing_point.shape[1] > 1:
            x_true, y_true = vanishing_point[index]
            x_pred, y_pred = predicted_point[index]
        elif vanishing_point.shape[1] == 1:
            x_true, y_true = vanishing_point
            x_pred, y_pred = predicted_point
        else:
            return 'There is no coords provided.'

        distance = np.linalg.norm([x_true - x_pred, y_true - y_pred])

        # This distance will be used later for give more weight to
        # accuracies of the points which are closer to the center of the image
        distance_from_center = np.linalg.norm([x_pred - 0.5, y_pred - 0.5])

        from_center_distances.append(distance_from_center)

        if distance > max_dist:
            distance = max_dist

        # I think here is the place where need change.
        acc = (1 - distance / np.sqrt(2)) * 100  # If distance > np.sqrt(2) then the acc is negative
        acc = min(100, max(acc, 0))
        accs.append(acc)

    # Here I have used the 1 / np.sqrt(dist) formula to smooth the difference between
    # the accuracies of the points which are inside and which are outside the image
    weights = [1 / (np.sqrt(dist) if dist != 0 else 1e-10) for dist in from_center_distances]
    # print(f'This are the {weights}')
    weights = [dist / sum(weights) for dist in weights]

    overall_accuracy = sum(a * w for a, w in zip(accs, weights))

    return round(overall_accuracy, 2)
