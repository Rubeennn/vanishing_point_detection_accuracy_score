from vp_utils.preprocess import extract_vp_coords_from_json, normalize_vp_coords
from vp_utils.accuracy import accuracy_score, accuracy_score_with_penalty
from vp_utils.read_img import read_img


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




# vanishing_point_detection/
# ├── process/
# │   └── process.py         # Main pipeline or workflow.
# ├── core/                  # Instead of vp_utils (more descriptive name)
# │   ├── accuracy.py        # Handles accuracy-related calculations.
# │   ├── preprocess.py      # Preprocessing functions for vanishing points and images.
# │   └── io.py              # For reading and writing images (previously read_img).
# ├── tests/                 # Test cases (if you want to include them).
# │   └── test_accuracy.py   # Unit tests for accuracy calculations.
# └── data/                  # Test dataset or input images (optional).
