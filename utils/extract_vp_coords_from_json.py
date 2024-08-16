import json
from typing import Union
from pathlib import Path
import numpy as np


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

    coords = data.get('coordinates', {})

    try:
        return  np.array(list(coords.values())).reshape(2, 2)

    except Exception as e:
        return e
