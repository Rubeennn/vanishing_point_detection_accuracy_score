import cv2
import numpy as np
import time
from new_placing_algo import one_vanishing_point, best_lines
# from main_server_new import two_line_intersection, find_k_and_b
# from main_vanishing_points import get_vanishing_points_test
# from vanishing_accuracy import metric, get_accuracy
import json
from vp_utils.accuracy import accuracy_score


def two_line_intersection(line1_params, line2_params):

    (k1, b1), (k2, b2) = line1_params, line2_params
    x = (b2 - b1) / (k1 - k2)
    y = k2 * x + b2
    return [x, y]

def new_placing_algo(img, mask, floor_path, vanishing_points):

    # img_path = r"f1/23.jpg"
    # mask_path = r"f3/23_mask.png"
    # wall_path = "wood1.png"
    # img = cv2.imread(img_path)
    height, width = img.shape[:2]
    # img_new = img.copy()
    # mask = cv2.imread(mask_path)[..., 0]
    wood = cv2.imread(floor_path)
    # wood = floor_artificial_shadow(wood)
    # wood = np.transpose(wood, (1, 0, 2))
    only_floor = (mask == 3) | (mask == 28)
    # print(only_floor.shape)
    only_floor_1 = only_floor.copy().astype(np.uint8)
    only_floor_3d = np.transpose(np.array([only_floor, only_floor, only_floor]), axes=(1, 2, 0))
    vanishing_points = np.array(vanishing_points).astype(np.int32)
    if len(vanishing_points.shape) == 1:
        result_mask_floor = mask == 3
        pts2 = one_vanishing_point(vanishing_points, result_mask_floor)
        pts1 = np.array([[0, 0], [0, wood.shape[1]], [wood.shape[0], 0], [wood.shape[0], wood.shape[1]]])
        matrix_new = cv2.getPerspectiveTransform(pts1.astype(np.float32), np.float32(np.array(pts2)))
        img_output = cv2.warpPerspective(wood, matrix_new, (width, height))
        img = img * (~only_floor_3d)
        img_output = img_output * only_floor_3d
        # cv2.imshow("", img_output)
        # cv2.waitKey(0)
        img = img + img_output
        return img
    # print("old vanishing points", np.array(vanishing_points1))
    # print("new van", vanishing_points)
    # print("abc", vanishing_points)
    van1 = vanishing_points[0]  # np.array([-61, 331])
    van2 = vanishing_points[1]  # np.array([829, 331])
    # print("van", van1)
    # print("van", van2)
    van1_left, van1_right = best_lines(img, only_floor_1, van1)
    # print(van1_left, van1_right)
    x1 = (height - van1_left[1]) / van1_left[0]
    x2 = (height - van1_right[1]) / van1_right[0]
    if x1 > x2:
        van1_left, van1_right = van1_right, van1_left
    van2_left, van2_right = best_lines(img, only_floor_1, van2)
    x1 = (height - van2_left[1]) / van2_left[0]
    x2 = (height - van2_right[1]) / van2_right[0]
    if x1 > x2:
        van2_left, van2_right = van2_right, van2_left
    # img, _ = draw_line(van1_left[0], van1_left[1], img, (255, 0, 0))
    # img, _ = draw_line(van1_right[0], van1_right[1], img, (0, 255, 0))
    # img, _ = draw_line(van2_left[0], van2_left[1], img, (0, 0, 255))
    # img, _ = draw_line(van2_right[0], van2_right[1], img, (255, 255, 255))
    # print(van1_left, van1_right)
    # print(van2_left, van2_right)
    # print("-"*100)
    pt1 = two_line_intersection(van1_right, van2_left)
    pt2 = two_line_intersection(van1_right, van2_right)
    pt3 = two_line_intersection(van1_left, van2_left)
    pt4 = two_line_intersection(van1_left, van2_right)
    pts1 = np.array([[0, 0], [0, wood.shape[1]], [wood.shape[0], 0], [wood.shape[0], wood.shape[1]]])
    # pts2 = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
    pts2 = np.array([pt1, pt2, pt3, pt4]).astype(np.int32)
    # print("pts1", pts1)
    # print("pts2", pts2)
    #
    # print(pts1.shape)
    matrix_new = cv2.getPerspectiveTransform(pts1.astype(np.float32), np.float32(np.array(pts2)))
    img_output = cv2.warpPerspective(wood, matrix_new, (width, height))
    # cv2.imshow("", img_output)
    # cv2.waitKey(0)
    # cv2.imwrite("test_output.jpg", img_output)
    img = img * (~only_floor_3d)
    img_output = img_output * only_floor_3d
    img = img + img_output
    return img

def track_func(x):
    pass

def scaling_image(img, mask=None, vanishing_points=None, scale=1):

    img_shape = (np.array(img.shape[:2]) * scale).astype(np.uint16)
    # print(tuple(img_shape))
    img = cv2.resize(img, img_shape[::-1], interpolation=cv2.INTER_NEAREST)
    if not mask is None:
        mask = cv2.resize(mask, img_shape[::-1], interpolation=cv2.INTER_NEAREST)
    if not vanishing_points is None:
        vanishing_points = vanishing_points * scale
    return img, mask, vanishing_points

def scaling_image_width(img, mask=None, vanishing_points=None, scaling_width=None):

    ratio = img.shape[0] / img.shape[1]
    scale = scaling_width / img.shape[1]
    img_shape = [scaling_width, round(ratio * scaling_width)]
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_NEAREST)
    if not mask is None:
        mask = cv2.resize(mask, img_shape, interpolation=cv2.INTER_NEAREST)
    if not vanishing_points is None:
        vanishing_points = vanishing_points * scale
    return img, mask, vanishing_points

def run(img_path, mask_path, wood_path, vanishing_points_old, metric):

    cv2.namedWindow("abc")
    cv2.createTrackbar("van1_x", "abc", 0, 500, track_func)
    cv2.createTrackbar("van1_y", "abc", 0, 200, track_func)
    cv2.createTrackbar("van2_x", "abc", 0, 500, track_func)
    cv2.createTrackbar("van2_y", "abc", 0, 200, track_func)
    cv2.setTrackbarMin("van1_x", "abc", -500)
    cv2.setTrackbarMin("van1_y", "abc", -200)
    cv2.setTrackbarMin("van2_x", "abc", -500)
    cv2.setTrackbarMin("van2_y", "abc", -200)
    mask = cv2.imread(mask_path)[..., 0]
    img = cv2.imread(img_path)
    width = img.shape[1]
    img, mask, vanishing_points_old = scaling_image_width(img, mask, vanishing_points_old, scaling_width=500)
    while True:
        # cv2.imshow("abc", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        van1_x = cv2.getTrackbarPos("van1_x", "abc")
        van1_y = cv2.getTrackbarPos("van1_y", "abc")
        van2_x = cv2.getTrackbarPos("van2_x", "abc")
        van2_y = cv2.getTrackbarPos("van2_y", "abc")
        vanishing_points = np.array([[van1_x, van1_y], [van2_x, van2_y]])
        vanishing_points = vanishing_points + vanishing_points_old
        # print(vanishing_points)
        # print(vanishing_points)
        cv2.imshow("abc", new_placing_algo(img, mask, wood_path, vanishing_points))
        print(metric(vanishing_points_old, vanishing_points))
    cv2.destroyAllWindows()

num = 344



img_path = f"../../../Downloads/vanishing_point_detection_testing_images_1/image_{num}.jpg"
mask_path = f"../../../Pictures/masks/{num}_mask.png"
wood_path = "../../../Pictures/tile807.png"




with open(f'test_data_1/vps/vp_true/vanishing_point_{num}_true.json', 'r') as json_file:
    data_list = json.load(json_file)

data_array = (np.array(list(data_list['vanishing_points'].values()))).reshape((2,2))
run(img_path, mask_path, wood_path, data_array, accuracy_score)


















