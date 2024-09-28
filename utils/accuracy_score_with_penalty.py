import numpy as np
from utils.accuracy_score import accuracy_score
from utils.extract_vp_coords_from_json import extract_vp_coords_from_json


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
    far = 0
    index = 0
    if vp_true_coords.shape[1] == 2:
        vp_pred_squeezed = vp_pred_coords.squeeze()
        for i in range(2):
            distance = np.linalg.norm(vp_pred_squeezed - vp_true_coords[i, :])
            if distance > far:
                far = distance
                index = i

        vp_true_filtered = vp_true_coords[1 - index, :]
        vp_true_filtered = np.expand_dims(vp_true_filtered, axis=-1)

        penalty_coord = vp_true_coords[index, :]
        # print(f'The penalty coords are: {penalty_coord}, close pred {vp_true_filtered}, true coord {vp_pred_coords}')
        penalty_distance = np.linalg.norm([penalty_coord[0] - 0.5, penalty_coord[1] - 0.5])
        amount = (1 - np.sqrt(2) / penalty_distance)
        print(f'The penalty: {amount}')
        initial_score = accuracy_score(vp_true_filtered, vp_pred_coords)
        initial_score = initial_score * (1 - amount)

        return initial_score

    elif vp_pred_coords.shape[1] == 2:
        vp_true_squeezed = vp_true_coords.squeeze()
        for i in range(2):
            distance = np.linalg.norm(vp_true_squeezed - vp_pred_coords[i, :])
            if distance > far:
                far = distance
                index = i

        vp_pred_filtered = vp_pred_coords[1 - index, :]
        vp_pred_filtered = np.expand_dims(vp_pred_filtered, axis=-1)

        penalty_coord = vp_pred_coords[index, :]
        # print(f'The penalty coords are: {penalty_coord}, close pred {vp_pred_filtered}, true coord {vp_true_coords}')

        penalty_distance = np.linalg.norm([penalty_coord[0] - 0.5, penalty_coord[1] - 0.5])
        # print(f'The penalty: {penalty_distance}')
        amount = (1 - np.sqrt(2) / penalty_distance)
        print(f'The penalty: {amount}')

        initial_score = accuracy_score(vp_true_coords, vp_pred_filtered)
        initial_score = initial_score * (1 - amount)
        return initial_score


vp_true = extract_vp_coords_from_json('../vps/vp_true/vanishing_point_13_true.json')
vp_pred = extract_vp_coords_from_json('../vps/vp_pred/vanishing_point_13_predicted.json')
# print(accuracy_score_with_penalty(vp_true, vp_pred))
