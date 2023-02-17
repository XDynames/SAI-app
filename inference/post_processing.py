import os
import json

import numpy as np
from scipy.stats import iqr

from tools.constants import (
    CLOSE_TO_EDGE_DISTANCE,
    CLOSE_TO_EDGE_SIZE_THRESHOLD,
    SIZE_THRESHOLD,
    IOU_THRESHOLD,
)

Predicted_Pore_Lengths, Bounding_Boxes = [], []


def get_indices_of_valid_predictions(predictions):
    remove_intersecting_predictions(predictions)
    i_away_from_edge = get_indices_of_detections_away_from_edges(predictions)
    i_above_minimum_size = get_indices_of_detections_above_minimum_size(
        predictions
    )
    valid_indices = i_away_from_edge.intersection(i_above_minimum_size)
    return valid_indices


def remove_intersecting_predictions(predictions):
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        intersecting = [
            j
            for j, bbox_j in enumerate(predictions.pred_boxes)
            if not i == j and is_overlapping(bbox_i, bbox_j)
        ]
        is_larger = [
            predictions.scores[i].item() > predictions.scores[j].item()
            for j in intersecting
        ]
        if not intersecting:
            final_indices.append(i)
        elif len(is_larger) > 0 and all(is_larger):
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_overlapping(bbox1, bbox2):
    if intersects(bbox1, bbox2):
        return overlaps(bbox1, bbox2)
    return False


def intersects(bbox_1, bbox_2):
    is_overlap = not (
        bbox_2[0] > bbox_1[2]
        or bbox_2[2] < bbox_1[0]
        or bbox_2[1] > bbox_1[3]
        or bbox_2[3] < bbox_1[1]
    )
    return is_overlap


def overlaps(bbox_1, bbox_2):
    iou = intersection_over_union(bbox_1, bbox_2)
    if iou > IOU_THRESHOLD:
        return True
    return False


def intersection_over_union(bbox, bbox_2):
    x_max, y_max = max(bbox[0], bbox_2[0]), max(bbox[1], bbox_2[1])
    x_min, y_min = min(bbox[2], bbox_2[2]), min(bbox[3], bbox_2[3])
    intersecting_area = max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)

    pred_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    gt_area = (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1)
    iou = intersecting_area / float(pred_area + gt_area - intersecting_area)
    return iou


def select_predictions(predictions, indices):
    predictions.pred_boxes.tensor = predictions.pred_boxes.tensor[indices]
    predictions.pred_classes = predictions.pred_classes[indices]
    predictions.pred_masks = predictions.pred_masks[indices]
    predictions.pred_keypoints = predictions.pred_keypoints[indices]
    predictions.scores = predictions.scores[indices]


def get_indices_of_detections_away_from_edges(predictions):
    average_area = calculate_average_bbox_area(predictions)
    image_height, image_width = predictions.image_size
    valid_indices = set()
    for i, bbox_i in enumerate(predictions.pred_boxes):
        is_near_edge = is_bbox_near_edge(bbox_i, image_height, image_width)
        is_significantly_smaller_than_average = is_bbox_small(
            bbox_i, average_area
        )
        if is_near_edge and is_significantly_smaller_than_average:
            continue
        else:
            valid_indices.add(i)
    return valid_indices


def calculate_average_bbox_area(predictions):
    areas = [calculate_bbox_area(bbox) for bbox in predictions.pred_boxes]
    if not areas:
        return 0
    return sum(areas) / len(areas)


def calculate_bbox_area(bbox):
    width = calculate_bbox_width(bbox)
    height = calculate_bbox_height(bbox)
    return width * height


def calculate_bbox_height(bbox):
    return abs(bbox[3] - bbox[1])


def calculate_bbox_width(bbox):
    return abs(bbox[2] - bbox[0])


def is_bbox_near_edge(bbox, image_height, image_width):
    x1, y1, x2, y2 = bbox
    is_near_edge = any(
        [
            x1 < CLOSE_TO_EDGE_DISTANCE,
            y1 < CLOSE_TO_EDGE_DISTANCE,
            image_width - x2 < CLOSE_TO_EDGE_DISTANCE,
            image_height - y2 < CLOSE_TO_EDGE_DISTANCE,
        ]
    )
    return is_near_edge


def is_bbox_small(bbox, average_area):
    threshold_area = CLOSE_TO_EDGE_SIZE_THRESHOLD * average_area
    bbox_area = calculate_bbox_area(bbox)
    return bbox_area < threshold_area


def get_indices_of_detections_above_minimum_size(predictions):
    average_area = calculate_average_bbox_area(predictions)
    valid_indices = set()
    for i, bbox_i in enumerate(predictions.pred_boxes):
        if is_bbox_extremley_small(bbox_i, average_area):
            continue
        else:
            valid_indices.add(i)
    return valid_indices


def is_bbox_extremley_small(bbox, average_area):
    return calculate_bbox_area(bbox) < average_area * SIZE_THRESHOLD


def remove_outliers_from_records():
    path = "./output/temp/"
    for filename in os.listdir(path):
        if ".json" in filename:
            if "-gt" in filename:
                continue
            filepath = os.path.join(path, filename)
            record = load_json(filepath)
            remove_outliers(record, filename)
            write_to_json(record, filepath)


def load_json(filepath):
    with open(filepath, "r") as file:
        record = json.load(file)
    record["valid_detection_indices"] = set(record["valid_detection_indices"])
    return record


def unpack_predictions(record):
    predictions = []
    for detection in record:
        predictions.append(detection["pred"])
    return predictions


def extract_pore_lengths(record):
    indices, lengths = [], []
    predictions = record["detections"]
    valid_indices = record["valid_detection_indices"]
    for i, prediction in enumerate(predictions):
        if i in valid_indices:
            indices.append(i)
            lengths.append(prediction["length"])
    return indices, lengths


def remove_outliers(record, filename):
    remove_pore_length_outliers(record)
    remove_bounding_box_outliers(record)


def remove_pore_length_outliers(record):
    indices, length_predictions = extract_pore_lengths(record)
    to_remove = find_outlier_indices(indices, length_predictions)
    for i in to_remove:
        record["valid_detection_indices"].remove(i)


def remove_bounding_box_outliers(record):
    near_edge_indices = get_indices_of_near_edge_detections(record)
    remove_bounding_box_height_outliers(record, near_edge_indices)
    remove_bounding_box_width_outliers(record, near_edge_indices)


def get_indices_of_near_edge_detections(record):
    near_edge_indices = []
    predictions = record["detections"]
    image_size = record["image_size"]
    for i, prediction in enumerate(predictions):
        if is_bbox_near_edge(prediction["bbox"], image_size[0], image_size[1]):
            near_edge_indices.append(i)
    return set(near_edge_indices)


def remove_bounding_box_height_outliers(record, near_edge_indices):
    heights = extract_bbox_heights(record)
    minimum, maximum = calculate_bbox_height_limits()
    to_remove = find_outlier_bbox_indices(
        heights,
        minimum,
        maximum,
        near_edge_indices,
    )
    remove_outlier_records(record, to_remove)


def remove_bounding_box_width_outliers(record, near_edge_indices):
    widths = extract_bbox_widths(record)
    minimum, maximum = calculate_bbox_width_limits()
    to_remove = find_outlier_bbox_indices(
        widths,
        minimum,
        maximum,
        near_edge_indices,
    )
    remove_outlier_records(record, to_remove)


def extract_bbox_heights(record):
    heights = []
    predictions = record["detections"]
    for prediction in predictions:
        heights.append(calculate_bbox_height(prediction["bbox"]))
    return heights


def extract_bbox_widths(record):
    widths = []
    predictions = record["detections"]
    for prediction in predictions:
        widths.append(calculate_bbox_width(prediction["bbox"]))
    return widths


def find_outlier_bbox_indices(
    sizes,
    lower_bound,
    upper_bound,
    near_edge_indices,
):
    to_remove = set()
    for i, size in enumerate(sizes):
        if i not in near_edge_indices:
            if (size < lower_bound) or (size > upper_bound):
                to_remove.add(i)
        else:
            if is_less_than_half_of_average_size(
                size,
                sizes,
                near_edge_indices,
            ):
                to_remove.add(i)
    return list(to_remove)


def is_less_than_half_of_average_size(size, sizes, near_edge_indices):
    valid_sizes = []
    for i, i_size in enumerate(sizes):
        if i not in near_edge_indices:
            valid_sizes.append(i_size)
    if len(valid_sizes) == 0:
        average_size = 0.0
    else:
        average_size = sum(valid_sizes) / len(valid_sizes)
    return size < (average_size * 0.5)


def calculate_bbox_height_limits():
    heights = [bbox["height"] for bbox in Bounding_Boxes]
    inter_quartile_range = iqr(heights, interpolation="midpoint")
    median = np.median(heights)
    lower_whisker = median - 2.0 * inter_quartile_range
    upper_whisker = median + 2.0 * inter_quartile_range
    return lower_whisker, upper_whisker


def calculate_bbox_width_limits():
    widths = [bbox["width"] for bbox in Bounding_Boxes]
    inter_quartile_range = iqr(widths, interpolation="midpoint")
    median = np.median(widths)
    lower_whisker = median - 2.0 * inter_quartile_range
    upper_whisker = median + 2.0 * inter_quartile_range
    return lower_whisker, upper_whisker


def find_outlier_indices(indices, lengths):
    to_remove = []
    lower_bound = calculate_length_limit()
    for i, length in enumerate(lengths):
        if length < lower_bound:
            to_remove.append(indices[i])
    return to_remove


def calculate_length_limit():
    inter_quartile_range = iqr(
        Predicted_Pore_Lengths, interpolation="midpoint"
    )
    median = np.median(Predicted_Pore_Lengths)
    lower_whisker = median - 2.0 * inter_quartile_range
    return lower_whisker


def write_to_json(record, filepath):
    record["valid_detection_indices"] = list(record["valid_detection_indices"])
    with open(filepath, "w") as file:
        json.dump(record, file)


def remove_outlier_records(record, to_remove):
    predictions = record["detections"]
    set_of_valid_indices = record["valid_detection_indices"]
    to_remove, valid_indices, i = set(to_remove), [], 0
    for j in range(len(predictions)):
        if (j in to_remove) and (j not in set_of_valid_indices):
            i -= 1
        elif j in set_of_valid_indices:
            valid_indices.append(i)
        i += 1
    for j in range(len(predictions))[::-1]:
        if j in to_remove and (j not in set_of_valid_indices):
            predictions.pop(j)
    record["detections"] = predictions
    record["valid_detection_indices"] = set(valid_indices)
