import os
import json

import cv2
import pandas as pd
import numpy as np
import shapely.geometry as shapes
import streamlit as st
from mask_to_polygons.vectorification import geometries_from_mask
from matplotlib import pyplot as plt
from PIL import Image
from shapely import affinity
from rasterio.transform import IDENTITY

from app import utils
from app.example_images import (
    draw_bounding_boxes,
    draw_measurements,
    filter_immature_stomata,
    filter_low_confidence_predictions,
    setup_plot,
)
from app.image_retrieval import get_selected_image
from app.summary_statistics import is_valid_calibration
from interface.upload_single import convert_to_SIU_length
from interface.upload_folder import (
    get_download_csv_button,
    show_save_visualisations_options,
    show_side_by_side_buttons,
)
from inference.infer import run_on_image
from inference.post_processing import (
    Bounding_Boxes,
    Predicted_Pore_Lengths,
    calculate_bbox_height,
    calculate_bbox_width,
    remove_outliers_from_records,
)

from tools.constants import (
    DENSITY_KEYS,
    DENSITY_OUTPUT_COLUMNS,
    MEASUREMENT_KEYS,
    MEASUREMENT_OUTPUT_COLUMN_NAMES,
    MINIMUM_LENGTH,
    OPENCV_FILE_SUPPORT,
    WIDTH_OVER_LENGTH_THRESHOLD,
)
from tools.load import (
    clean_temporary_folder,
    maybe_create_visualisation_folder,
)
from tools.state import Option_State


def maybe_do_inference():
    if utils.is_file_uploaded() and utils.is_mode_upload_an_example():
        maybe_do_single_image_inference()
    if utils.is_image_folder_avaiable() and utils.is_mode_upload_multiple_images():
        maybe_do_inference_on_all_images_in_folder()


def maybe_do_single_image_inference():
    if not is_inference_available_for_uploaded_image():
        with st.spinner("Measuring Stomata..."):
            image = get_selected_image()
            predictions, time_elapsed, valid_indices = run_on_image(image)
            predictions, valid_indices = predictions_to_list_of_dictionaries(
                predictions, valid_indices
            )
            Option_State["uploaded_inference"] = {
                "name": Option_State["uploaded_file"]["name"],
                "model_used": Option_State["plant_type"],
                "predictions": predictions,
                "valid_detection_indices": valid_indices,
            }
        st.success(f"Finished in {time_elapsed:2f}s")


def is_inference_available_for_uploaded_image():
    if not is_upload_inference_available():
        return False
    check = is_inference_done_using_the_selected_model()
    check = check and is_inference_for_the_current_file()
    return check


def is_upload_inference_available():
    return not Option_State["uploaded_inference"] is None


def is_inference_for_the_current_file():
    currently_uploaded_filename = Option_State["uploaded_file"]["name"]
    filename_of_available_inference = Option_State["uploaded_inference"]["name"]
    return currently_uploaded_filename == filename_of_available_inference


def is_inference_done_using_the_selected_model():
    currently_selected_model = Option_State["plant_type"]
    model_used_for_inference = Option_State["uploaded_inference"]["model_used"]
    return currently_selected_model == model_used_for_inference


def maybe_do_inference_on_all_images_in_folder():
    if not is_inference_available_for_folder():
        if is_infer_button_pressed():
            clean_temporary_folder()
            reset_tracked_predictions()
            do_inference_on_all_images_in_folder()
    if is_inference_available_for_folder():
        reset_predictions()
        apply_user_settings()
        density_filename, measurement_filename = create_output_csvs()
        display_download_links(density_filename, measurement_filename)
        display_visualisation_options()
        maybe_visualise_and_save()


def is_inference_available_for_folder():
    if not is_folder_inference_available():
        return False
    check = is_folder_inference_done_using_the_selected_model()
    check = check and is_inference_for_the_current_folder()
    return check


def is_folder_inference_available():
    return not Option_State["folder_inference"] is None


def is_folder_inference_done_using_the_selected_model():
    currently_selected_model = Option_State["plant_type"]
    model_used_for_inference = Option_State["folder_inference"]["model_used"]
    return currently_selected_model == model_used_for_inference


def is_inference_for_the_current_folder():
    current_folder = Option_State["folder_path"]
    available_inference = Option_State["folder_inference"]["name"]
    return current_folder == available_inference


def is_infer_button_pressed():
    pressed = Option_State["infer_button"]
    Option_State["infer_button"] = False
    return pressed


def reset_tracked_predictions():
    Predicted_Pore_Lengths = []
    Bounding_Boxes = []


def do_inference_on_all_images_in_folder():
    total_time = 0
    directory = Option_State["folder_path"]
    image_files = get_list_of_images_in_folder(directory)
    if len(image_files) == 0:
        return
    progress = 0
    progress_bar_header = st.empty()
    with progress_bar_header:
        st.write(f"Measuring {len(image_files)} images...")
    progress_bar = st.progress(progress)
    increment = 100 / len(image_files)
    status_container = st.empty()

    n_stoma = 0
    for filename in image_files:
        image = np.array(Image.open(f"{directory}/{filename}"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        predictions, time_elapsed, valid_indices = run_on_image(image)
        record_predictions(
            predictions,
            filename,
            n_stoma,
            valid_indices,
            image.shape,
        )

        progress += increment
        progress_bar.progress(int(progress))

        with status_container:
            st.info(f"{filename} completed in {time_elapsed:.2f}s")

        total_time += time_elapsed
        n_stoma += len(predictions)

    progress_bar.progress(100)
    progress_bar.empty()
    progress_bar_header.empty()
    with status_container:
        st.success(f"Measured {len(image_files)} images in {total_time:.2f}s")

    remove_outliers_from_records()
    Option_State["folder_inference"] = {
        "name": Option_State["folder_path"],
        "model_used": Option_State["plant_type"],
        "predictions": load_all_saved_predictions(),
    }


def get_list_of_images_in_folder(folder_path):
    filenames = os.listdir(folder_path)
    image_files = [
        filename for filename in filenames if is_supported_image_file(filename)
    ]
    return image_files


def is_supported_image_file(filename):
    return filename.split(".")[-1] in OPENCV_FILE_SUPPORT


def store_bounding_boxes(predictions):
    for prediction in predictions:
        Bounding_Boxes.append(
            {
                "height": calculate_bbox_height(prediction["bbox"]),
                "width": calculate_bbox_width(prediction["bbox"]),
            }
        )


def record_predictions(
    predictions,
    filename,
    n_stoma,
    valid_indices,
    image_size,
):
    path = "./output/temp/"
    predictions, valid_indices = predictions_to_list_of_dictionaries(
        predictions, valid_indices, n_stoma
    )
    store_bounding_boxes(predictions)
    filename = remove_extension_from_filename(filename)
    to_save = {
        "detections": predictions,
        "valid_detection_indices": list(valid_indices),
        "image_size": image_size[:-1],
    }
    with open(path + ".".join([filename, "json"]), "w") as file:
        json.dump(to_save, file)


def remove_extension_from_filename(filename):
    file_extension = "." + filename.split(".")[-1]
    return filename.replace(file_extension, "")


def predictions_to_list_of_dictionaries(predictions, valid_indices, n_stoma=0):
    predictions = [
        predictions_to_dictionary(i, predictions, n_stoma, valid_indices)
        for i in range(len(predictions.pred_boxes))
    ]
    return predictions, valid_indices


def predictions_to_dictionary(i, predictions, n_stoma, valid_indices):
    # Extract and format predictions
    pred_mask = predictions.pred_masks[i].cpu().numpy()
    pred_AB = predictions.pred_keypoints[i].flatten().tolist()
    pred_class = predictions.pred_classes[i].item()
    pred_length = l2_dist(pred_AB)

    # Processes prediction mask
    if pred_class == 1:
        pred_polygon = mask_to_poly(pred_mask)
        # Sanity check for keypoint prediction
        if pred_length < MINIMUM_LENGTH and pred_polygon:
            pred_AB = get_AB_from_polygon(pred_polygon)
            pred_length = l2_dist(pred_AB)

        pred_CD = find_CD(pred_polygon, pred_AB)
        # Retry using polygon keypoints
        if pred_CD == [-1, -1, 1, -1, -1, 1] and pred_polygon:
            pred_AB = get_AB_from_polygon(pred_polygon)
            pred_CD = find_CD(pred_polygon, pred_AB)
            pred_length = l2_dist(pred_AB)

        width_length_ratio = calulate_width_over_length(pred_length, l2_dist(pred_CD))
        if width_length_ratio > WIDTH_OVER_LENGTH_THRESHOLD:
            pred_AB = get_AB_from_polygon(pred_polygon)
            pred_CD = find_CD(pred_polygon, pred_AB)
            pred_length = l2_dist(pred_AB)

        pred_width = l2_dist(pred_CD)
        pred_area = pred_mask.sum().item()

    else:
        pred_polygon = []
        pred_CD = [-1, -1, 1.0 - 1, -1, 1]
        pred_width = 0
        pred_area = 0
    # Stoma length is always the longest measurement
    if pred_width > pred_length:
        pred_AB, pred_CD = pred_CD, pred_AB
        pred_length, pred_width = pred_width, pred_length

    width_length_ratio = calulate_width_over_length(pred_length, pred_width)
    if width_length_ratio > WIDTH_OVER_LENGTH_THRESHOLD:
        if i in valid_indices:
            valid_indices.remove(i)

    prediction_dict = {
        "stoma_id": i + n_stoma,
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
    if i in valid_indices:
        Predicted_Pore_Lengths.append(pred_length)
    return prediction_dict


def get_AB_from_polygon(polygon):
    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]
    return extract_polygon_AB(x_points, y_points)


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
            polygon = shapes.Polygon(polygon["coordinates"][0])
        except Exception:
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

    if keypoints is None:
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


def reset_predictions():
    Option_State["folder_inference"]["predictions"] = load_all_saved_predictions()


def apply_user_settings():
    apply_confidence_filter()
    apply_immature_stoma_filter()
    maybe_convert_length_measurement_units()


def apply_confidence_filter():
    apply_function(filter_low_confidence_predictions)


def apply_immature_stoma_filter():
    apply_function(filter_immature_stomata)


def apply_function(function):
    predictions = Option_State["folder_inference"]["predictions"]
    for i, image in enumerate(predictions):
        image_predictions = image["detections"]
        image_predictions = function(image_predictions)
        predictions[i]["detections"] = image_predictions


def maybe_convert_length_measurement_units():
    if is_valid_calibration():
        apply_function(convert_measurements)


def convert_measurements(predictions):
    for prediction in predictions:
        prediction["width"] = convert_to_SIU_length(prediction["width"])
        prediction["length"] = convert_to_SIU_length(prediction["length"])
        prediction["area"] = convert_to_SIU_length(prediction["area"])
        prediction["area"] = convert_to_SIU_length(prediction["area"])
    return predictions


def create_output_csvs():
    predictions = Option_State["folder_inference"]["predictions"]
    valid_predictions = format_predictions(predictions)
    measurement_csv_filename = write_measurements_to_csv(valid_predictions)
    densities = format_densities(predictions)
    density_csv_filename = write_density_csv(densities)
    return density_csv_filename, measurement_csv_filename


def load_all_saved_predictions():
    directory = "./output/temp/"
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
        valid_indices = prediction["valid_detection_indices"]
        for i, detection in enumerate(detections):
            if i not in valid_indices:
                continue
            measurements = {
                "stoma_id": detection["stoma_id"],
                "width": detection["width"],
                "length": detection["length"],
                "area": detection["area"],
                "class": "open" if detection["category_id"] else "closed",
                "confidence": detection["confidence"],
                "image_name": image_name,
                "width/length": calculate_width_to_length_ratio(detection),
            }
            stoma_measurements.append(measurements)
    return stoma_measurements


def calculate_width_to_length_ratio(detection):
    return calulate_width_over_length(detection["length"], detection["width"])


def calulate_width_over_length(length, width):
    return width / length if length > 0 else 0


def write_measurements_to_csv(measurements):
    measurement_csv_filepath = write_output_csv(
        measurements,
        MEASUREMENT_OUTPUT_COLUMN_NAMES,
        MEASUREMENT_KEYS,
        "pore_measurements",
    )
    return measurement_csv_filepath


def write_output_csv(measurements, column_names, column_keys, name):
    csv = ",".join(column_names) + "\n"
    for measurement in measurements:
        values = []
        for key in column_keys:
            values.append(measurement[key])
        values = [str(x) for x in values]
        csv += ",".join(values) + "\n"

    path = Option_State["folder_path"]
    if os.path.basename(path) == "":
        directory_name = os.path.basename(os.path.dirname(path))
    else:
        directory_name = os.path.basename(path)
    filename = f"{name}_{directory_name}.csv"
    with open(f"./output/temp/{filename}", "w") as file:
        file.write(csv)
    return filename


def format_densities(predictions):
    densities = []
    for prediction in predictions:
        detections = prediction["detections"]
        area = calculate_image_area(prediction["image_size"])
        n_stomata = len(detections)
        density = {
            "image_name": prediction["image_name"],
            "n_stomata": n_stomata,
            "density": n_stomata / area,
        }
        densities.append(density)
    return densities


def calculate_image_area(image_size):
    height = convert_to_SIU_length(image_size[0])
    width = convert_to_SIU_length(image_size[1])
    if Option_State["camera_calibration"] > 0:
        height, width = height / 1000, width / 1000
    return height * width


def write_density_csv(densities):
    density_csv_filepath = write_output_csv(
        densities,
        DENSITY_OUTPUT_COLUMNS,
        DENSITY_KEYS,
        "density",
    )
    return density_csv_filepath


def display_download_links(density_csv_name, measurement_csv_name):
    measurement_df = pd.read_csv("./output/temp/" + measurement_csv_name)
    measurement_button = get_download_csv_button(
        measurement_df,
        measurement_csv_name,
        "Download Pore Measurements",
    )
    density_df = pd.read_csv("./output/temp/" + density_csv_name)
    density_button = get_download_csv_button(
        density_df,
        density_csv_name,
        "Download Density Measurements",
    )
    show_side_by_side_buttons(measurement_button, density_button)


def display_visualisation_options():
    show_save_visualisations_options()


def maybe_visualise_and_save():
    if Option_State["visualise"]:
        maybe_create_visualisation_folder()
        visualise_and_save()
        Option_State["visualise"] = False


def visualise_and_save():
    visualisation_folder = Option_State["visualisation_path"]
    image_folder = Option_State["folder_path"]
    image_names = get_list_of_images_in_folder(image_folder)

    status_container = st.empty()
    progress = 0
    progress_bar_header = st.empty()
    with progress_bar_header:
        st.write(f"Visualising {len(image_names)} images...")
    progress_bar = st.progress(progress)
    increment = 100 / len(image_names)

    # Was working now stops execution in draw_and_save_visualisation
    #   at image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # with futures.ProcessPoolExecutor() as executor:
    #    future_to_image_name = {
    #        executor.submit(draw_and_save_visualisation, image_name):
    #        image_name for image_name in image_names
    #    }
    #    for result in futures.as_completed(future_to_image_name):
    #        progress += increment
    #        progress_bar.progress(int(progress))

    for image_name in image_names:
        draw_and_save_visualisation(image_name)
        progress += increment
        progress_bar.progress(int(progress))

    progress_bar.progress(100)
    progress_bar.empty()
    progress_bar_header.empty()
    with status_container:
        st.success(
            f"{len(image_names)} visualised images are now saved in {visualisation_folder}"
        )


def draw_and_save_visualisation(image_name):
    output_path = Option_State["visualisation_path"]
    input_path = Option_State["folder_path"]
    # Load image
    image = cv2.imread(os.path.join(input_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create matplot lib axis
    fig, ax = setup_plot(image)
    # Load images measurements
    predictions_filename = image_name.split(".")[0] + ".json"
    prediction_path = os.path.join("./output/temp/", predictions_filename)
    record = utils.load_json(prediction_path)
    predictions = record["detections"]
    valid_indices = record["valid_detection_indices"]
    valid_predictions = utils.select_predictions(predictions, valid_indices)
    # Draw onto axis
    draw_measurements(ax, valid_predictions)
    draw_bounding_boxes(ax, predictions)
    # Save drawing
    fig.savefig(
        os.path.join(output_path, image_name),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
