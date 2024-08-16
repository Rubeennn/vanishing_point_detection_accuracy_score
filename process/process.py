from typing import Union
from pathlib import Path
import json
from utils.accuracy_score import accuracy_score
from utils.extract_vp_coords_from_json import extract_vp_coords_from_json
from utils.generate_synthetic_vanishing_points_with_noise import (
    generate_synthetic_vanishing_points_with_noise)
from utils.normalize_vp_coords import normalize_vp_coords
from utils.read_img import read_img


def process(image_path: Union[str, Path],
            vp_true_json: json):

    vp_true = extract_vp_coords_from_json(vp_true_json)

    img = read_img(img_path=image_path)

    synthetic_prediction = generate_synthetic_vanishing_points_with_noise(vp_true)

    vp_true = normalize_vp_coords(vp_coordinates=vp_true,
                                  img_shape=img.shape)

    synthetic_prediction = normalize_vp_coords(vp_coordinates=synthetic_prediction,
                                               img_shape=img.shape)

    overall_accuracy = accuracy_score(vanishing_point=vp_true,
                                      predicted_point=synthetic_prediction)

    return overall_accuracy


# data = np.array([[-7090, 181], [350, 500]])
# predicted_points = np.array([[-10000, 340], [340, 420]])

print(process(image_path='../../../../Downloads/2023_10_24_09_46_29_954.jpg',
              vp_true_json='../../../../Downloads/2023_10_24_09_46_29_954.json'))


