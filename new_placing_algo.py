import numpy as np
import cv2
# from place_tile_count import place_tile_count
# from compute_area import in_point_area
# from new_lsd_1 import return_vanishing_points
# from use_lines import new_vanishing_points
# from horizontal_lines import detect_line_wall
# from main_server_new import two_line_intersection, find_k_and_b
# from shadows_lighting import floor_artificial_shadow
# from main_vanishing_points import get_vanishing_points_test
# from new_lsd_1 import draw_line


def two_line_intersection(line1_params, line2_params):
    (k1, b1), (k2, b2) = line1_params, line2_params
    x = (b2 - b1) / (k1 - k2)
    y = k2 * x + b2
    return [x, y]


def find_k_and_b(first_point, second_point):
    first_point = list(first_point)
    second_point = list(second_point)
    if first_point[0] == second_point[0]:
        first_point[0] -= 1
    k = (first_point[1] - second_point[1]) / (first_point[0] - second_point[0])
    b = -second_point[0] * (first_point[1] - second_point[1]) / (first_point[0] - second_point[0]) + second_point[1]
    return [k, b]

def extract_new_points(floor_segment):
    vertical_point_count = np.array([np.sum(floor_segment[row]) for row in range(floor_segment.shape[0])])
    horizontal_point_count = np.array([np.sum(floor_segment[:, column]) for column in range(floor_segment.shape[1])])
    min_height = np.argwhere(vertical_point_count != 0)[0]


    min_width, max_width = np.argwhere(horizontal_point_count != 0)[[0, -1]]
    min_width = min_width - 1 if min_width > 0 else min_width
    max_width = max_width + 1 if max_width < floor_segment.shape[1] else max_width
    min_x = np.argwhere(floor_segment[min_height] != 0)[0, 1]
    min_height[0] = min_height[0] - 2
    left_point = [min_width[0], min_height[0]]
    right_point = [max_width[0], min_height[0]]
    max_point = [min_x, min_height[0]]
    return left_point, right_point, max_point


def one_vanishing_point(vanishing_point, floor_segment):
    left_point, right_point, max_point = extract_new_points(floor_segment)
    # print(left_point, right_point)

    # vanishing_point = [floor_segment.shape[1] // 2, left_point[1] - 200]
    left_line = find_k_and_b(vanishing_point, left_point)
    right_line = find_k_and_b(vanishing_point, right_point)
    right_line_inter = [(floor_segment.shape[0] - right_line[1]) / right_line[0], floor_segment.shape[0]]
    left_line_inter = [(floor_segment.shape[0] - left_line[1]) / left_line[0], floor_segment.shape[0]]
    pts = [left_point, right_point, left_line_inter, right_line_inter]

    return pts


def best_lines(img, wall, vanishing_point):
    # print("vanish", vanishing_point)
    contours, hierarchy = cv2.findContours(wall,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    new_contours = []
    for u in range(len(contours)):
        new_contours.extend(np.array(contours[u][:, 0]))

    img_copy = img.copy()
    # for i in range(len(new_contours) - 1):
    #     img = cv2.line(img, new_contours[i], new_contours[i+1], color=(0, 0, 255), thickness=3)
    new_contours = np.array(new_contours)
    new_contours = new_contours[:, None]
    vanishing_point = np.array(vanishing_point)[None, None, :]
    new_contours = np.append(new_contours, vanishing_point, axis=0)
    # print("new countours", new_contours.shape)
    # print(new_contours)
    new_contours = new_contours.astype(np.int32)
    hull = cv2.convexHull(new_contours)
    vanish_index = np.all((hull[:, 0] == vanishing_point)[0], axis=1)
    vanish_index = np.argwhere(vanish_index)[0, 0]
    vanish_index = np.array([vanish_index-1, vanish_index, vanish_index+1])
    vanish_index = vanish_index % len(hull)
    new_points = hull[vanish_index][:, 0]
    # print("new points", new_points)
    line1 = find_k_and_b(new_points[0], new_points[1])
    line2 = find_k_and_b(new_points[1], new_points[2])
    if line1[0] > line2[0]:
        line1, line2 = line2, line1

    # point1 = [0, ]

    # print("hull shape", hull.shape)
    # for i in range(len(hull) - 1):
    #     img_copy = cv2.line(img_copy, hull[i, 0], hull[i+1, 0], color=(0, 0, 255), thickness=3)
    # img_copy = cv2.line(img_copy, hull[-1, 0], hull[0, 0], color=(0, 0, 255), thickness=3)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    # cv2.imshow('Contours', img_copy)
    # cv2.waitKey(0)

    # cv2.imwrite("tst.png", img)
    # cv2.imwrite("tst_hull.png", img_copy)
    return line1, line2


def get_new_4_points(img_path, mask_path, floor_path, vanishing_points=None):
    # img_path = r"f1/23.jpg"
    # mask_path = r"f3/23_mask.png"
    # wall_path = "wood1.png"
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    # img_new = img.copy()
    mask = cv2.imread(mask_path)[..., 0]
    # print(img.shape)
    # print(mask.shape)
    wood = cv2.imread(floor_path)

    # wood = wood[:2000, :2000]

    warp_width = wood.shape[1]
    warp_height = wood.shape[0]
    # wood = floor_artificial_shadow(wood)
    wood = np.transpose(wood, (1, 0, 2))
    only_floor = (mask == 3) | (mask == 28)
    only_floor_1 = only_floor.copy().astype(np.uint8)
    only_floor_3d = np.transpose(np.array([only_floor, only_floor, only_floor]), axes=(1, 2, 0))
    # vanishing_points1, _ = return_vanishing_points(img_path, mask)

    # vanishing_points, _ = new_vanishing_points(img_path, mask)
    if vanishing_points is None:
        vanishing_points, _ = get_vanishing_points_test(img_path, mask, "True")
        vanishing_points = np.array(vanishing_points).astype(np.int32)
    print("v", vanishing_points)
    print("vanish1", vanishing_points.shape)

    if len(vanishing_points.shape) == 1:
        result_mask_floor = mask == 3
        pts2 = one_vanishing_point(vanishing_points, result_mask_floor)
        print("pts2", pts2)
        pts2 = np.array(pts2)

        points_new = np.array([pts2[0], pts2[1], pts2[3], pts2[2]])
        in_point_area(points_new, img)
        wood = place_tile_count(img, wood, pts2)

        pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
        # pts1 = np.array([[0, 0], [0, wood.shape[1]], [wood.shape[0], 0], [wood.shape[0], wood.shape[1]]])
        # print("pts1", pts1)
        # print("pts2", pts2)
        #
        # print(pts1.astype(np.float32))
        # print(np.float32(np.array(pts2)))
        matrix_new = cv2.getPerspectiveTransform(np.array(pts1), np.float32(np.array(pts2)))
        return matrix_new, wood, pts1

    # if len(vanishing_points.shape) == 1:
    #     van1 = vanishing_points
    #     van1 = np.array(van1, dtype=np.float32)
    #     van1_left, van1_right = best_lines(img, only_floor_1, van1)
    #     # print(van1_left, van1_right)
    #     x1 = (height - van1_left[1]) / van1_left[0]
    #     x2 = (height - van1_right[1]) / van1_right[0]
    #     if x1 > x2:
    #         van1_left, van1_right = van1_right, van1_left
    #
    #     y = np.argwhere(np.sum(only_floor_1, axis=1) > 0)[0][0]
    #
    #     pt1 = np.array(two_line_intersection(van1_left, [0, y]))
    #     pt2 = np.array(two_line_intersection(van1_right, [0, y]))
    #     pt3 = np.array(two_line_intersection(van1_left, [0, img.shape[0]]))
    #     pt4 = np.array(two_line_intersection(van1_right, [0, img.shape[0]]))
    #
    #     pts2 = np.array([pt1, pt2, pt3, pt4]).astype(np.int32)
    #     wood = place_tile_count(wood, pts2)
    #     warp_width = wood.shape[1]
    #     warp_height = wood.shape[0]
    #     pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
    #     matrix_new = cv2.getPerspectiveTransform(np.array(pts1), np.float32(np.array(pts2)))
    #
    #     return matrix_new, wood, pts1

    # print("dfajhdfjahsklfja")
    # print("old vanishing points", np.array(vanishing_points1))
    # print("new van", vanishing_points)
    # print("abc", vanishing_points)
    vanishing_points = vanishing_points.astype(np.int32)
    van1 = vanishing_points[0]#np.array([-61, 331])
    van2 = vanishing_points[1]#np.array([829, 331])
    # print("van", van1)
    # print("van", van2)

    van1_left, van1_right = best_lines(img, only_floor_1, van1)
    # print(van1_left, van1_right)
    x1 = (height - van1_left[1])/van1_left[0]
    x2 = (height - van1_right[1])/van1_right[0]
    if x1 > x2:
        van1_left, van1_right = van1_right, van1_left
    van2_left, van2_right = best_lines(img, only_floor_1, van2)
    x1 = (height - van2_left[1])/van2_left[0]
    x2 = (height - van2_right[1])/van2_right[0]
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
    # pts1 = np.array([[0, 0], [0, wood.shape[1]], [wood.shape[0], 0], [wood.shape[0], wood.shape[1]]])
    pts2 = np.array([pt1, pt2, pt3, pt4]).astype(np.int32)
    wood = place_tile_count(img, wood, pts2)
    warp_width = wood.shape[1]
    warp_height = wood.shape[0]
    pts1 = np.float32([[0, 0], [warp_width, 0], [0, warp_height], [warp_width, warp_height]])
    # pts2 = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])


    # print("pts1", pts1)
    #
    # print(pts1.shape)
    matrix_new = cv2.getPerspectiveTransform(pts1.astype(np.float32), np.float32(np.array(pts2)))
    pts1 = pts1.astype(np.int32)
    print(matrix_new)
    print(wood.shape)
    print(pts1)

    return matrix_new, wood, pts1


def new_placing_algo(img_path, mask_path, floor_path, output_path, vanishing_points=None):
    # img_path = r"f1/23.jpg"
    # mask_path = r"f3/0.png"
    # wall_path = "wood1.png"
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    # img_new = img.copy()
    mask = cv2.imread(mask_path)[..., 0]
    # print(img.shape)
    # print(mask.shape)
    wood = cv2.imread(floor_path)
    # wood = floor_artificial_shadow(wood)
    # wood = np.transpose(wood, (1, 0, 2))
    only_floor = (mask == 3) | (mask == 28)
    only_floor_1 = only_floor.copy().astype(np.uint8)
    only_floor_3d = np.transpose(np.array([only_floor, only_floor, only_floor]), axes=(1, 2, 0))
    # vanishing_points1, _ = return_vanishing_points(img_path, mask)

    # vanishing_points, _ = new_vanishing_points(img_path, mask)
    if vanishing_points is None:
        vanishing_points, _ = get_vanishing_points_test(img_path, mask, "True")
        vanishing_points = np.array(vanishing_points).astype(np.int32)
    # print("vanish1", vanishing_points.shape)
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
        # cv2.imwrite(output_path, img)
        return img


    # print("old vanishing points", np.array(vanishing_points1))
    # print("new van", vanishing_points)
    # print("abc", vanishing_points)
    van1 = vanishing_points[0]#np.array([-61, 331])
    van2 = vanishing_points[1]#np.array([829, 331])
    # print("van", van1)
    # print("van", van2)
    van1_left, van1_right = best_lines(img, only_floor_1, van1)
    # print(van1_left, van1_right)
    x1 = (height - van1_left[1])/van1_left[0]
    x2 = (height - van1_right[1])/van1_right[0]
    if x1 > x2:
        van1_left, van1_right = van1_right, van1_left
    van2_left, van2_right = best_lines(img, only_floor_1, van2)
    x1 = (height - van2_left[1])/van2_left[0]
    x2 = (height - van2_right[1])/van2_right[0]
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

    # print(pts1.shape)
    matrix_new = cv2.getPerspectiveTransform(pts1.astype(np.float32), np.float32(np.array(pts2)))
    img_output = cv2.warpPerspective(wood, matrix_new, (width, height))
    # cv2.imshow("", img_output)
    # cv2.waitKey(0)
    # cv2.imwrite("test_output.jpg", img_output)
    img = img * (~only_floor_3d)
    img_output = img_output * only_floor_3d
    img = img + img_output

    # cv2.imwrite("test_new_placing.png", img_output)
    # cv2.imshow("", img_output)
    # cv2.waitKey(0)
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # cv2.imwrite(output_path, img)
    return img


# new_placing_algo(r"f1\40.jpg", r"f3\40_mask.png", "wood1.png")
# cv2.imshow("", new_placing_algo(r"f1/30.jpg", r"f3/30_mask.png", "wood1.png", "test_place_1.jpg"))
# cv2.waitKey(0)

