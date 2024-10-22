from typing import Union, Optional
from pathlib import Path
import cv2 as cv
import numpy as np


def read_img(img_path: Union[str, Path]) -> Optional[np.array]:
    """Reads the image using its path.

    Parameters:
        img_path (Union[str, Path]): The path of the image.

    Returns:
        Optional[np.array]: The image as a numpy array.
        """
    try:
        # Convert Path to str if necessary
        if isinstance(img_path, Path):
            img_path = str(img_path)

        # Attempt to read the image
        img = cv.imread(img_path)

        # Check if the image was loaded successfully
        if img is None:
            raise ValueError("Image not found or unable to load")
        return img

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



# img_path = "../../../../Downloads/203_10_24_09_46_29_954.jpg"
#
# # Call the function
# img = read_img(img_path)