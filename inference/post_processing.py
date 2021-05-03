import streamlit as st
from inference.record import is_overlapping

CLOSE_TO_EDGE_DISTANCE = 20
CLOSE_TO_EDGE_SIZE_THRESHOLD = 0.7
SIZE_THRESHOLD = 0.3


def filter_invalid_predictions(predictions):
    remove_intersecting_predictions(predictions)
    remove_close_to_edge_detections(predictions)
    remove_extremley_small_detections(predictions)


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


def select_predictions(predictions, indices):
    predictions.pred_boxes.tensor = predictions.pred_boxes.tensor[indices]
    predictions.pred_classes = predictions.pred_classes[indices]
    predictions.pred_masks = predictions.pred_masks[indices]
    predictions.pred_keypoints = predictions.pred_keypoints[indices]
    predictions.scores = predictions.scores[indices]


def remove_close_to_edge_detections(predictions):
    average_area = calculate_average_bbox_area(predictions)
    image_height,  image_width  = predictions.image_size
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        is_near_edge = is_bbox_near_edge(bbox_i, image_height, image_width)
        is_significantly_smaller_than_average = is_bbox_small(bbox_i, average_area)
        if is_bbox_near_edge and is_significantly_smaller_than_average:
            continue
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def calculate_average_bbox_area(predictions):
    areas = [ calculate_bbox_area(bbox) for bbox in predictions.pred_boxes ]
    if not areas:
        return 0
    return sum(areas) / len(areas)


def calculate_bbox_area(bbox):
    width =  abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    return width * height


def is_bbox_near_edge(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox    
    is_near_edge = any([
        x1 < CLOSE_TO_EDGE_DISTANCE,
        y1 < CLOSE_TO_EDGE_DISTANCE,
        image_width - x2 < CLOSE_TO_EDGE_DISTANCE,
        image_height - y2 < CLOSE_TO_EDGE_DISTANCE,
    ])
    return is_near_edge

def is_bbox_small(bbox, average_area):
    threshold_area =  CLOSE_TO_EDGE_SIZE_THRESHOLD * average_area
    bbox_area = calculate_bbox_area(bbox)
    return bbox_area < threshold_area

def remove_extremley_small_detections(predictions):
    average_area = calculate_average_bbox_area(predictions)
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        if is_bbox_extremley_small(bbox_i, average_area):
            continue
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_bbox_extremley_small(bbox, average_area):
    return calculate_bbox_area(bbox) < average_area * SIZE_THRESHOLD