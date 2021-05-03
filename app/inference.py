import time

import numpy as np

import shapely
from shapely import affinity
import shapely.geometry as shapes
import streamlit as st
from rasterio.transform import IDENTITY
from mask_to_polygons.vectorification import geometries_from_mask

from app import utils
from app.image_retrieval import get_selected_image
from inference.infer import run_on_image
from app.example_images import setup_plot, draw_example
from tools.state import Option_State


def maybe_do_inference():
    if utils.is_file_uploaded() and utils.is_mode_upload_an_example():
        if not is_inference_available_for_uploaded_image():
            with st.spinner("Measuring Stomata..."):
                predictions, time_elapsed = run_on_image(get_selected_image())
                predictions = predictions_to_list_of_dictionaries(predictions)
                Option_State["uploaded_inference"] = {
                    'name': Option_State['uploaded_file']['name'],
                    'model_used': Option_State['plant_type'],
                    'predictions': predictions,
                }
            st.success(f"Finished in {time_elapsed:2f}s")


def is_inference_available_for_uploaded_image():
    if not is_inference_available():
        return False
    check = is_inference_done_using_the_selected_model()
    check = check and is_inference_for_the_current_file()
    return check

def is_inference_available():
    return not Option_State['uploaded_inference'] is None


def is_inference_for_the_current_file():
    currently_uploaded_filename = Option_State['uploaded_file']['name']
    filename_of_available_inference = Option_State['uploaded_inference']['name']
    return currently_uploaded_filename == filename_of_available_inference

def is_inference_done_using_the_selected_model():
    currently_selected_model = Option_State['plant_type']
    model_used_for_inference = Option_State['uploaded_inference']['model_used']
    return currently_selected_model == model_used_for_inference


def predictions_to_list_of_dictionaries(predictions):
    predictions = [
        predictions_to_dictionary(i, predictions)
        for i in range(len(predictions.pred_boxes))
    ]
    return predictions


def predictions_to_dictionary(i, predictions):
    # Extract and format predictions
    pred_mask = predictions.pred_masks[i].cpu().numpy()
    pred_AB = predictions.pred_keypoints[i].flatten().tolist()
    pred_class = predictions.pred_classes[i].item()
    if pred_class == 1:
        # Processes prediction mask
        pred_polygon = mask_to_poly(pred_mask)
        pred_CD = find_CD(pred_polygon, gt=False)
        pred_width = l2_dist(pred_CD)
        pred_area = pred_mask.sum().item()
    else:
        pred_polygon = []
        pred_CD = [-1, -1, 1.0 - 1, -1, 1]
        pred_width = 0
        pred_area = 0

    prediction_dict = {
        "bbox": predictions.pred_boxes[i].tensor.tolist()[0],
        "area": pred_area,
        "AB_keypoints": pred_AB,
        "length": l2_dist(pred_AB),
        "CD_keypoints": pred_CD,
        "width": pred_width,
        "category_id": pred_class,
        "segmentation": [pred_polygon],
        "confidence": predictions.scores[i].item(),
    }
    return prediction_dict


def mask_to_poly(mask):
    if mask.sum() == 0:
        return []
    poly = geometries_from_mask(np.uint8(mask), IDENTITY, "polygons")
    poly = poly[0]["coordinates"][0]
    flat_poly = []
    for point in poly:
        flat_poly.extend([point[0], point[1]])
    return flat_poly


def find_CD(polygon, keypoints=None, gt=True):
    # If no mask is predicted
    if len(polygon) < 1:
        #    counter += 1
        return [-1, -1, 1, -1, -1, 1]

    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]

    if keypoints == None:
        keypoints = extract_polygon_AB(x_points, y_points)
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
    except:
        intersections = shapes.collection.GeometryCollection()
    # If there is no intersection
    if intersections.is_empty:
        return [-1, -1, 1, -1, -1, 1]

    if intersections[0].coords.xy[1] > intersections[1].coords.xy[1]:
        D = intersections[0].coords.xy
        C = intersections[1].coords.xy
    else:
        D = intersections[1].coords.xy
        C = intersections[0].coords.xy
    return [*C[0], *C[1], 1, *D[0], *D[1], 1]


def extract_polygon_AB(x_values, y_values):
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


def l2_dist(keypoints):
    A, B = [keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)