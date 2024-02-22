from typing import List, Union

import streamlit as st
import shapely.geometry as shapes
from shapely import affinity

from inference.constants import IOU_THRESHOLD, NAMES_TO_CATEGORY_ID


def is_stomatal_pore(i, predictions) -> bool:
    class_label = get_class(i, predictions)
    return class_label == NAMES_TO_CATEGORY_ID["Stomatal Pore"]


def is_stomata_complex(i, predictions) -> bool:
    class_label = get_class(i, predictions)
    complex_categories = [
        NAMES_TO_CATEGORY_ID["Closed Stomata"],
        NAMES_TO_CATEGORY_ID["Open Stomata"],
    ]
    return class_label in complex_categories


def get_class(i, predictions) -> int:
    return predictions.pred_classes[i].item()


def get_bounding_box(i, predictions) -> List[float]:
    return predictions.pred_boxes[i].tensor.tolist()[0]


def intersects(bbox_1, bbox_2) -> bool:
    is_overlap = not (
        bbox_2[0] > bbox_1[2]
        or bbox_2[2] < bbox_1[0]
        or bbox_2[1] > bbox_1[3]
        or bbox_2[3] < bbox_1[1]
    )
    return is_overlap


def is_overlapping(bbox1, bbox2) -> bool:
    if intersects(bbox1, bbox2):
        return is_overlap(bbox1, bbox2, IOU_THRESHOLD)
    return False


def is_overlap(bbox_1, bbox_2, threshold) -> bool:
    iou = intersection_over_union(bbox_1, bbox_2)
    if iou > threshold:
        return True
    return False


def intersection_over_union(bbox, gt_bbox):
    intersecting_area = calculate_area_of_intersection(bbox, gt_bbox)
    pred_area = calculate_area_of_bbox(bbox)
    gt_area = calculate_area_of_bbox(gt_bbox)
    iou = intersecting_area / float(pred_area + gt_area - intersecting_area)
    return iou


def calculate_area_of_intersection(bbox, gt_bbox) -> float:
    x_max, y_max = max(bbox[0], gt_bbox[0]), max(bbox[1], gt_bbox[1])
    x_min, y_min = min(bbox[2], gt_bbox[2]), min(bbox[3], gt_bbox[3])
    return max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)


def calculate_area_of_bbox(bbox) -> float:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def is_bbox_a_in_bbox_b(a: List[float], b: List[float]) -> bool:
    # Checks if a bounding box in xyxy format is contained within another
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def is_bbox_a_mostly_in_bbox_b(
    bbox_a: List[float],
    bbox_b: List[float],
    threshold: float,
) -> bool:
    intersecting_area = calculate_area_of_intersection(bbox_a, bbox_b)
    bbox_a_area = calculate_area_of_bbox(bbox_a)
    return intersecting_area / bbox_a_area > threshold


def l2_dist(keypoints):
    A, B = [keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)


def extract_AB_from_polygon(
    x_values: List[float],
    y_values: List[float],
) -> List[float]:
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    # Enables pores of arbitrary orientation
    if x_extent > y_extent:
        major_axis_values = x_values
        minor_axis_values = y_values
        maximum_major_value = x_max
        minimum_major_value = x_min
    else:
        major_axis_values = y_values
        minor_axis_values = x_values
        maximum_major_value = y_max
        minimum_major_value = y_min
    # Left/Right along major axis
    left_hand_values, right_hand_values = [], []
    for i, minor_value in enumerate(minor_axis_values):
        if maximum_major_value == major_axis_values[i]:
            right_hand_values.append(minor_value)
        if minimum_major_value == major_axis_values[i]:
            left_hand_values.append(minor_value)
    # Use midpoint of extreme values as keypoint value
    right_hand_value = (right_hand_values[0] + right_hand_values[-1]) / 2
    left_hand_value = (left_hand_values[0] + left_hand_values[-1]) / 2

    if x_extent > y_extent:
        keypoints = [
            minimum_major_value,
            left_hand_value,
            1,
            maximum_major_value,
            right_hand_value,
            1,
        ]
    else:
        keypoints = [
            left_hand_value,
            minimum_major_value,
            1,
            right_hand_value,
            maximum_major_value,
            1,
        ]
    return keypoints


def find_CD(polygon, keypoints=None):
    # If no mask is predicted
    if len(polygon) < 1:
        #    counter += 1
        return [-1, -1, 1, -1, -1, 1]

    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]

    if keypoints is None:
        keypoints = extract_AB_from_polygon(x_points, y_points)
    # Convert to shapely linear ring
    polygon = [[x, y] for x, y in zip(x_points, y_points)]
    mask = shapes.LinearRing(polygon)
    # Find line perpendicular to AB
    A = shapes.Point(keypoints[0], keypoints[1])
    B = shapes.Point(keypoints[3], keypoints[4])

    l_AB = shapes.LineString([A, B])
    l_perp = affinity.rotate(l_AB, 90)
    l_perp = affinity.scale(l_perp, 10, 10)
    # Find intersection with polygon
    try:
        intersections = l_perp.intersection(mask)
    except Exception:
        intersections = shapes.collection.GeometryCollection()
    # If there is no intersection or only one point of intersection
    if intersections.is_empty or type(intersections) is shapes.Point:
        return [-1, -1, 1, -1, -1, 1]
    # If there are multiple intersections, pick the largest
    if len(intersections.geoms) > 2:
        intersections = select_longest_line(intersections)

    if intersections.geoms[0].coords.xy[1] > intersections.geoms[1].coords.xy[1]:
        D = intersections.geoms[0].coords.xy
        C = intersections.geoms[1].coords.xy
    else:
        D = intersections.geoms[1].coords.xy
        C = intersections.geoms[0].coords.xy
    return [C[0][0], C[1][0], 1, D[0][0], D[1][0], 1]


def select_longest_line(multipoint):
    lines, lengths = [], []
    for i, point_1 in enumerate(multipoint.geoms):
        for point_2 in multipoint.geoms[i + 1 :].geoms:
            lines.append(shapes.LineString([point_1, point_2]))
            lengths.append(lines[-1].length)
    longest_line_idx = max(range(len(lengths)), key=lambda i: lengths[i])
    longest_line = lines[longest_line_idx]
    return shapes.MultiPoint(list(longest_line.coords))


def calulate_width_over_length(length, width):
    return width / length if length > 0 else 0


def calculate_bbox_area(bbox):
    width = calculate_bbox_width(bbox)
    height = calculate_bbox_height(bbox)
    return width * height


def calculate_bbox_height(bbox):
    return abs(bbox[3] - bbox[1])


def calculate_bbox_width(bbox):
    return abs(bbox[2] - bbox[0])
