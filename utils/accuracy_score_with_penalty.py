import numpy as np
from utils.accuracy_score import accuracy_score
from utils.extract_vp_coords_from_json import extract_vp_coords_from_json


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
        initial_score = initial_score * (1 - penalty)

        return round(initial_score, 2)

    elif vp_pred_coords.shape[1] == 2:
        vp_pred_filtered, penalty = calculate_penalty(coord_2d=vp_pred_coords, coord_1d=vp_true_coords)

        initial_score = accuracy_score(vp_true_coords, vp_pred_filtered)
        initial_score = initial_score * (1 - penalty)
        return round(initial_score, 2)


vp_true = extract_vp_coords_from_json('../vps/vp_true/vanishing_point_13_true.json')
vp_pred = extract_vp_coords_from_json('../vps/vp_pred/vanishing_point_13_predicted.json')
# print(accuracy_score_with_penalty(vp_true, vp_pred))
