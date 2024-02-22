from typing import List

from inference.utils import (
    is_bbox_a_in_bbox_b,
    is_bbox_a_mostly_in_bbox_b,
    is_stomata_complex,
    is_stomatal_pore,
    is_overlapping,
    get_bounding_box,
)
from inference.constants import (
    CLOSE_TO_EDGE_DISTANCE,
    CLOSE_TO_EDGE_SIZE_THRESHOLD,
    ORPHAN_AREA_THRESHOLD,
    SIZE_THRESHOLD,
)


def filter_invalid_predictions(predictions):
    remove_intersecting_predictions(predictions)
    remove_close_to_edge_detections(predictions)
    remove_extremely_small_detections(predictions)
    remove_orphan_detections(predictions)


def remove_intersecting_predictions(predictions):
    final_indices = []
    for i in range(len(predictions.pred_boxes)):
        if is_stomata_complex(i, predictions):
            if is_best_prediction(i, predictions):
                final_indices.append(i)
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_best_prediction(i: int, predictions) -> bool:
    intersecting = get_intersecting_prediction_indices(i, predictions)
    is_larger = is_score_larger_intersecting_predictions(i, predictions, intersecting)
    return all(is_larger) or not intersecting


def get_intersecting_prediction_indices(i: int, predictions) -> List[int]:
    bbox = get_bounding_box(i, predictions)
    intersecting = [
        j
        for j, bbox_j in enumerate(predictions.pred_boxes)
        if not i == j and is_overlapping(bbox, bbox_j)
    ]
    return intersecting


def is_score_larger_intersecting_predictions(
    i: int,
    predictions,
    intersecting: List[int],
) -> List[bool]:
    is_larger = [
        predictions.scores[i].item() >= predictions.scores[j].item()
        for j in intersecting
    ]
    return is_larger


def select_predictions(predictions, indices: List[int]):
    predictions.pred_boxes.tensor = predictions.pred_boxes.tensor[indices]
    predictions.pred_classes = predictions.pred_classes[indices]
    predictions.pred_masks = predictions.pred_masks[indices]
    predictions.pred_keypoints = predictions.pred_keypoints[indices]


def remove_close_to_edge_detections(predictions):
    final_indices = []
    average_area = calculate_average_bbox_area(predictions)
    for i in range(len(predictions.pred_boxes)):
        if is_stomata_complex(i, predictions):
            if is_detection_too_close_to_edge(i, predictions, average_area):
                continue
            else:
                final_indices.append(i)
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def calculate_average_bbox_area(predictions) -> float:
    areas = [
        calculate_bbox_area(bbox)
        for i, bbox in enumerate(predictions.pred_boxes)
        if is_stomata_complex(i, predictions)
    ]
    n_bbox = len(areas)
    if n_bbox == 0:
        return 0
    return sum(areas) / n_bbox


def is_detection_too_close_to_edge(i, predictions, average_area: float) -> bool:
    image_height, image_width = predictions.image_size
    bbox = get_bounding_box(i, predictions)
    is_near_edge = is_bbox_near_edge(bbox, image_height, image_width)
    is_significantly_smaller_than_average = is_bbox_small(bbox, average_area)
    return is_near_edge and is_significantly_smaller_than_average


def is_bbox_near_edge(bbox: List[float], image_height: int, image_width: int) -> bool:
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


def is_bbox_small(bbox: List[float], average_area: float) -> bool:
    threshold_area = CLOSE_TO_EDGE_SIZE_THRESHOLD * average_area
    bbox_area = calculate_bbox_area(bbox)
    return bbox_area < threshold_area


def calculate_bbox_area(bbox: List[float]) -> float:
    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    return width * height


def remove_extremely_small_detections(predictions):
    final_indices = []
    average_area = calculate_average_bbox_area(predictions)
    for i, bbox_i in enumerate(predictions.pred_boxes):
        if is_stomata_complex(i, predictions):
            if is_bbox_extremely_small(bbox_i, average_area):
                continue
            else:
                final_indices.append(i)
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_bbox_extremely_small(bbox, average_area):
    return calculate_bbox_area(bbox) < average_area * SIZE_THRESHOLD


def remove_orphan_detections(predictions):
    final_indices = []
    for i in range(len(predictions.pred_boxes)):
        if not is_stomata_complex(i, predictions):
            bbox = get_bounding_box(i, predictions)
            if is_orphan(i, bbox, predictions):
                continue
            else:
                final_indices.append(i)
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_orphan(i, bbox, predictions) -> bool:
    for j in range(len(predictions.pred_boxes)):
        if is_stomata_complex(j, predictions):
            stomata_bbox = get_bounding_box(j, predictions)
            if is_stomatal_pore(i, predictions):
                if is_bbox_a_in_bbox_b(bbox, stomata_bbox):
                    return False
            else:
                if _is_bbox_a_mostly_in_bbox_b(bbox, stomata_bbox):
                    return False
    return True


def _is_bbox_a_mostly_in_bbox_b(bbox_a, bbox_b):
    return is_bbox_a_mostly_in_bbox_b(bbox_a, bbox_b, ORPHAN_AREA_THRESHOLD)