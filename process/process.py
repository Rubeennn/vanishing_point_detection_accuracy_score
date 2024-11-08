from vp_utils.preprocess import extract_vp_coords_from_json, normalize_vp_coords
from vp_utils.accuracy import accuracy_score, accuracy_score_with_penalty
from vp_utils.read_img import read_img
from pathlib import Path
import numpy as np

def process(true_vp_json, pred_vp_json, image_path):
    vp_true = extract_vp_coords_from_json(true_vp_json)
    vp_pred = extract_vp_coords_from_json(pred_vp_json)

    img = read_img(img_path=image_path)

    vp_true = normalize_vp_coords(vp_coordinates=vp_true,
                                  img_shape=img.shape)

    vp_pred = normalize_vp_coords(vp_coordinates=vp_pred,
                                  img_shape=img.shape)

    if vp_pred.shape == vp_true.shape:
        overall_accuracy = accuracy_score(vanishing_point=vp_true,
                                          predicted_point=vp_pred)
    else:
        print(f'Need penalty for the image No: {image_path}')
        overall_accuracy = accuracy_score_with_penalty(vp_true_coords=vp_true,
                                                       vp_pred_coords=vp_pred)

    return overall_accuracy


# num = 777
# true_json = Path(f'../test_data_1/vps/vp_true/vanishing_point_{num}_true.json')
# pred_json = Path(f'../test_data_1/vps/vp_pred_new/vanishing_point_{num}_predicted.json')
# image = Path(f'../test_data_1/images/image_{num}.jpg')
# print(process(true_json, pred_json, image))

# accuracy_results = {i: 0.0 for i in [9,10,11,12,13,14,15,16,17,18,344,539,704,711,722,777,789,1056,1151]}
# for index in accuracy_results.keys():
#     try:
#
#         true_json = Path(f'../test_data_1/vps/vp_true/vanishing_point_{index}_true.json')
#         pred_json = Path(f'../test_data_1/vps/vp_pred_new/vanishing_point_{index}_predicted.json')
#         image = Path(f'../test_data_1/images/image_{index}.jpg')
#
#         accuracy_results[index] = process(true_json, pred_json, image)
#     except Exception as e:
#         print(f'the error accured for the image number {index}: {e}')
#
#
# for key, value in accuracy_results.items():
#     # print(f"{key}: {value}")
#     print(value)
#
# print(sum(accuracy_results.values()) / 16)


# num = 1151
# true_json = Path(f'../test_data_2/vps/vp_true/vanishing_point_{num}_true.json')
# pred_json = Path(f'../test_data_2/vps/vp_pred/vanishing_point_{num}_predicted.json')
# image = Path(f'../test_data_2/images/image_{num}.png')
# print(process(true_json, pred_json, image))

# accuracy_results = {i: 0.0 for i in [51,149,160,165,170,201,207,293,294,350,366,369,412,449,456,473,478,999]}
# for index in accuracy_results.keys():
#     try:
#
#         true_json = Path(f'../test_data_2/vps/vp_true/vanishing_point_{index}_true.json')
#         pred_json = Path(f'../test_data_2/vps/vp_pred_new/vanishing_point_{index}_predicted.json')
#         image = Path(f'../test_data_2/images/image_{index}.png')
#
#         accuracy_results[index] = process(true_json, pred_json, image)
#     except Exception as e:
#         print(f'the error accured for the image number {index}: {e}')
#
# for key, value in accuracy_results.items():
#     # print(f"{key}: {value}")
#     print(value)

# vanishing_point_detection/
# ├── process/
# │   └── process.py         # Main pipeline or workflow.
# ├── core/                  # Instead of vp_utils (more descriptive name)
# │   ├── accuracy.py        # Handles accuracy-related calculations.
# │   ├── preprocess.py      # Preprocessing functions for vanishing points and images.
# │   └── io.py              # For reading and writing images (previously read_img).
# ├── tests/                 # Test cases (if you want to include them).
# │   └── test_accuracy.py   # Unit tests for accuracy calculations.
# └── test_data_1/                  # Test dataset or input images (optional).


# arr1 = np.array([[-2659.08, 200.94], [201.36, 95.98]])
# arr2 = np.array([ [2088, 218], [138, 218]])
# print(arr1.shape)
# print(np.sort(arr1, axis=0))
# print(np.sort(arr2, axis=0))