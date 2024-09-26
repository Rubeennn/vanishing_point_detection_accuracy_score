from utils.accuracy_score import accuracy_score
from utils.accuracy_score_with_penalty import accuracy_score_with_penalty
from utils.extract_vp_coords_from_json import extract_vp_coords_from_json
from utils.normalize_vp_coords import normalize_vp_coords
from utils.read_img import read_img
from pathlib import Path


# image_path = '../../../../Downloads/20/image_15.jpg'

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


# num = 10
# true_json = Path(f'../vps/vp_true/vanishing_point_{num}_true.json')
# pred_json = Path(f'../vps/vp_pred/vanishing_point_{num}_predicted.json')
# image = Path(f'../../../../Downloads/vanishing_point_detection_testing_images/image_{num}.jpg')
# print(process(true_json, pred_json, image))

accuracy_results = {i: 0.0 for i in [1,9,10,11,12,13,14,15,16,17,18,344,539,704,711,722,777,789,1056,1151]}
for index in accuracy_results.keys():
    try:

        true_json = Path(f'../vps/vp_true/vanishing_point_{index}_true.json')
        pred_json = Path(f'../vps/vp_pred/vanishing_point_{index}_predicted.json')
        image = Path(f'../../../../Downloads/vanishing_point_detection_testing_images/image_{index}.jpg')

        accuracy_results[index] = process(true_json, pred_json, image)
    except Exception as e:
        print(f'the error accured for the image number {index}: {e}')


for key, value in accuracy_results.items():
    print(f"{key}: {value}")

print(sum(accuracy_results.values()) / 16)
