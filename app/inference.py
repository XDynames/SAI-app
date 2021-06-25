import base64
import os
import json

import cv2
import pandas as pd
import numpy as np
import shapely.geometry as shapes
import streamlit as st
from mask_to_polygons.vectorification import geometries_from_mask
from shapely import affinity
from rasterio.transform import IDENTITY

from app import utils
from app.image_retrieval import get_selected_image
from inference.infer import run_on_image
from inference.post_processing import remove_outliers_from_records, Predicted_Lengths
from app.example_images import setup_plot, draw_example
from tools.state import Option_State
from tools.constants import OPENCV_FILE_SUPPORT

MINIMUM_LENGTH = 5

def maybe_do_inference():
    if utils.is_file_uploaded() and utils.is_mode_upload_an_example():
        maybe_do_single_image_inference()
    if utils.is_image_folder_avaiable() and utils.is_mode_upload_multiple_images():
        maybe_do_inference_on_all_images_in_folder()

def maybe_do_single_image_inference():
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


def maybe_do_inference_on_all_images_in_folder():
    total_time = 0
    Predicted_Lengths = []
    directory = Option_State['folder_path']
    filenames = os.listdir(directory)
    image_files = [
        filename for filename in filenames
        if is_supported_image_file(filename)
    ]
    progress = 0
    st.write(f"Measuring {len(image_files)} images...")
    progress_bar = st.progress(progress)
    increment  = 100 // len(image_files)
    status_container = st.empty()

    for filename in image_files:
        image = cv2.imread(directory + '/' + filename)
        predictions, time_elapsed = run_on_image(image)
        record_predictions(predictions, filename)
        progress += increment
        progress_bar.progress(progress)
        with status_container:
            st.info(f"{filename} completed in {time_elapsed:.2f}s")
        total_time += time_elapsed
    with status_container:
        st.success(f"Measured {len(image_files)} images in {total_time:.2f}s")
    remove_outliers_from_records()
    saved_filename = create_output_csv()
    display_download_link(saved_filename)


def is_supported_image_file(filename):
    return filename.split('.')[-1] in OPENCV_FILE_SUPPORT


def record_predictions(predictions, filename):
    path = './output/temp/'
    predictions = predictions_to_list_of_dictionaries(predictions)
    with open(path + ".".join([filename[:-4], "json"]), "w") as file:
        json.dump({"detections": predictions}, file)


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
        pred_CD = find_CD(pred_polygon)
        pred_width = l2_dist(pred_CD)
        pred_area = pred_mask.sum().item()
    else:
        pred_polygon = []
        pred_CD = [-1, -1, 1.0 - 1, -1, 1]
        pred_width = 0
        pred_area = 0
    
    pred_length = l2_dist(pred_AB)
    if pred_length < MINIMUM_LENGTH and pred_polygon:
        x_points = [x for x in pred_polygon[0::2]]
        y_points = [y for y in pred_polygon[1::2]]
        pred_AB = extract_polygon_AB(x_points, y_points)
        pred_length = l2_dist(pred_AB)

    prediction_dict = {
        "bbox": predictions.pred_boxes[i].tensor.tolist()[0],
        "area": pred_area,
        "AB_keypoints": pred_AB,
        "length": pred_length,
        "CD_keypoints": pred_CD,
        "width": pred_width,
        "category_id": pred_class,
        "segmentation": [pred_polygon],
        "confidence": predictions.scores[i].item(),
    }
    Predicted_Lengths.append(pred_length)
    return prediction_dict


def mask_to_poly(mask):
    if mask.sum() == 0:
        return []
    poly = geometries_from_mask(np.uint8(mask), IDENTITY, "polygons")
    if len(poly) > 1:
        poly = find_maximum_area_polygon(poly)
    else:
        poly = poly[0]
    poly = poly["coordinates"][0]
    return flatten_polygon(poly)


def find_maximum_area_polygon(polygons):
    maximum = 0
    index = 0
    for i, polygon in enumerate(polygons):
        try:
            polygon = shapely.geometry.Polygon(polygon['coordinates'][0])
        except:
            continue
        if polygon.area > maximum:
            maximum = polygon.area
            index = i
    return polygons[index]

def flatten_polygon(polygon):
    flat_polygon = []
    for point in polygon:
        flat_polygon.extend([point[0], point[1]])
    return flat_polygon


def find_CD(polygon, keypoints=None):
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


def create_output_csv():
    predictions = load_all_saved_predictions()
    predictions = format_predictions(predictions)
    return write_to_csv(predictions)


def load_all_saved_predictions():
    directory = './output/temp/'
    files = os.listdir(directory)
    image_detections = []
    for file in files:
        if not file[-4:] == "json":
            continue
        with open(os.path.join(directory, file), "r") as read_file:
            image_data = json.load(read_file)
            image_data["image_name"] = file[:-5]
            image_detections.append(image_data)
    return image_detections


def format_predictions(predictions):
    stoma_measurements = []
    for prediction in predictions:
        image_name = prediction["image_name"]
        detections = prediction["detections"]
        for detection in detections:
            measurements = {
                "width": detection["width"],
                "length": detection["length"],
                "area": detection["area"],
                "class": 'open' if detection["category_id"] else 'closed',
                "confidence": detection["confidence"],
                "image_name": image_name,
            }
            stoma_measurements.append(measurements)
    return stoma_measurements

def write_to_csv(measurments):
    column_names = [
        "id",
        "image_name",
        "pred_class",
        "pred_length",
        "pred_width",
        "pred_area",
        "confidence",
    ]
    column_keys = ["image_name", "class", "length", "width", "area", "confidence"]
    csv = ",".join(column_names) + "\n"

    for i, measurment in enumerate(measurments):
        values = [i]
        for stoma_property in column_keys:
            values.append(measurment[stoma_property])
        values = [str(x) for x in values]
        csv += ",".join(values) + "\n"
    
    path = Option_State['folder_path']
    if os.path.basename(path) == '':
        directory_name = os.path.basename(os.path.dirname(path))
    else:
        directory_name = os.path.basename(path)
    filename = f'{directory_name}.csv'
    with open(f'./output/temp/{filename}', "w") as file:
        file.write(csv)
    return filename


def display_download_link(temp_csv_name):
    dataframe = pd.read_csv('./output/temp/' + temp_csv_name)
    coded_data = base64.b64encode(dataframe.to_csv(index=False).encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{coded_data}" download="{temp_csv_name}">Download Measurements</a>', unsafe_allow_html=True)