from typing import Union
from pathlib import Path
import numpy as np
import json


def extract_vp_coords_from_json(json_path: Union[Path, str]) -> np.array:
    """Reads the  json file and returns the vanishing point coordinates as np.array.

    Parameters:
        json_path (Union[Path, str]): The path of the file.

    Returns:
        np.array: the vanishing points of the image."""

    if isinstance(json_path, Path):
        json_path = str(json_path)

    with open(json_path) as f:
        data = json.load(f)

    coords = data.get('vanishing_points', {})
    if isinstance(coords, dict):
        try:
            return  np.array(list(coords.values())).reshape(2, -1)

        except Exception as e:
            return e
    elif isinstance(coords, list):
        try:
            return np.array(coords).reshape(2,-1)
        except Exception as e:
            return  e


def normalize_vp_coords(vp_coordinates: np.array,
                        img_shape: tuple) -> np.array:
    """Normalizes the vanishing point coordinates to the range [0, 1].

    Parameters:
    vp_coordinates (np.array): The coordinates of the vanishing points.
    It should be a 2D array where each row represents the point.

    img_shape (tuple): The shape of the image.

    Returns:
    np.array: The normalized vanishing point coordinates with values in the range [0, 1]."""

    try:
        vp_coordinates = vp_coordinates.astype(float)

        vp_coordinates[:, 0] /= img_shape[0]
        if vp_coordinates.shape[1] == 2:
            vp_coordinates[:, 1] /= img_shape[1]
        return vp_coordinates
    except Exception as e:
        return e
