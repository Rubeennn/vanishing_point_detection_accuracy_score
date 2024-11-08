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

    vanishing_point = np.sort(vanishing_point, axis=0)
    predicted_point = np.sort(predicted_point, axis=0)
    print(f'the sorted true vanishing point: {vanishing_point}')
    print(f'the sorted pred vanishing point: {predicted_point}')

    for index in range(len(vanishing_point)):
        if vanishing_point.shape[1] > 1:
            x_true, y_true = vanishing_point[index]
            x_pred, y_pred = predicted_point[index]
        elif vanishing_point.shape[1] == 1:
            x_true, y_true = vanishing_point
            x_pred, y_pred = predicted_point
        else:
            return 'There is no coords provided.'

        distance = np.linalg.norm([(x_true - x_pred), (y_true - y_pred)])
        # weighted_dist = np.linalg.norm([(x_true - x_pred), 2*(y_true - y_pred)])
        # This distance will be used later for give more weight to
        # accuracies of the points which are closer to the center of the image
        distance_from_center = np.linalg.norm([x_true - 0.5, y_true - 0.5])

        from_center_distances.append(distance_from_center)

        if distance > max_dist:
            distance = max_dist

        # I think here is the place where need change.
        acc = (1 - distance / np.sqrt(2)) * 100  # If distance > np.sqrt(2) then the acc is negative
        acc = max(min(100, max(acc, 0)), 30)
        accs.append(acc)

    # Here I have used the 1 / np.sqrt(dist) formula to smooth the difference between
    # the accuracies of the points which are inside and which are outside the image
    weights = [1 / (np.sqrt(dist) if dist != 0 else 1e-10) for dist in from_center_distances]
    # print(f'This are the {weights}')
    weights = [dist / sum(weights) for dist in weights]

    overall_accuracy = sum(a * w for a, w in zip(accs, weights))

    return round(overall_accuracy, 2)


def calculate_penalty(coord_2d, coord_1d):
    """This function calculates the penalty.

    Parameters:
    coord_2d (np.array): The array with 2 vanishing points.
    coord_1d (np.array): The array with 1 vanishing point.

    Returns:
    The penalty and the filtered vanishing point coordinate.

        """
    farthest_distance = 0
    index = 0
    coord_1d_squeezed = coord_1d.squeeze()
    for i in range(2):
        distance = np.linalg.norm(coord_1d_squeezed - coord_2d[i, :])
        if distance > farthest_distance:
            farthest_distance = distance
            index = i

    coord_2d_filtered = coord_2d[1 - index, :]
    coord_2d_filtered = np.expand_dims(coord_2d_filtered, axis=-1)

    penalty_coord = coord_2d[index, :]
    penalty_distance = np.linalg.norm([penalty_coord[0] - 0.5, penalty_coord[1] - 0.5])
    amount = (1 - np.sqrt(2) / penalty_distance)
    return coord_2d_filtered, amount


def accuracy_score_with_penalty(vp_true_coords: np.array,
                                vp_pred_coords: np.array) -> float:
    """This function calculates the accuracy of the vanishing point detection, for the case
    where there are other vanishing points that do not have pairs, and therefore we need
    to add a penalty term to overall accuracy.

    Parameters:
    vp_true_coords (np.array): The true vanishing point coordinates.
    vp_pred_coords (np.array): The predicted vanishing point coordinates.

    Returns:
    The accuracy of detection with the account of the penalty.

        """

    if vp_true_coords.shape[1] == 2:
        vp_true_filtered, penalty = calculate_penalty(coord_2d=vp_true_coords, coord_1d=vp_pred_coords)

        print(f'The penalty: {penalty}')
        initial_score = accuracy_score(vp_true_filtered, vp_pred_coords)
        # initial_score = initial_score * (1 - penalty)

        return round(initial_score, 2)

    elif vp_pred_coords.shape[1] == 2:
        vp_pred_filtered, penalty = calculate_penalty(coord_2d=vp_pred_coords, coord_1d=vp_true_coords)

        initial_score = accuracy_score(vp_true_coords, vp_pred_filtered)
        # initial_score = initial_score * (1 - penalty)
        return round(initial_score, 2)
