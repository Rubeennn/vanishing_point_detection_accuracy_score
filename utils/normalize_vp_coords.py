import numpy as np


def normalize_vp_coords(vp_coordinates: np.array,
                        img_shape: tuple) -> np.array:
    """Normalizes the vanishing point coordinates to the range [0, 1].

    Parameters:
    vp_coordinates (np.array): The coordinates of the vanishing points.
    It should be a 2D array where each row represents the point.

    img_shape (tuple): The shape of the image.

    Returns:
    np.array: The normalized vanishing point coordinates with values in the range [0, 1].
    """
    try:
        vp_coordinates = vp_coordinates.astype(float)

        vp_coordinates[:, 0] /= img_shape[0]
        if vp_coordinates.shape[1] == 2:
            vp_coordinates[:, 1] /= img_shape[1]
        return vp_coordinates
    except Exception as e:
        return e

