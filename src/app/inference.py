import os
import json

import cv2
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

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
from inference.population_filtering import (
    Bounding_Boxes,
    Predicted_Pore_Lengths,
    remove_outliers_from_records,
)

from tools.constants import (
    DENSITY_KEYS,
    DENSITY_OUTPUT_COLUMNS,
    MEASUREMENT_KEYS,
    MEASUREMENT_OUTPUT_COLUMN_NAMES,
    OPENCV_FILE_SUPPORT,
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
            predictions, time_elapsed = run_on_image(image)
            Option_State["uploaded_inference"] = {
                "name": Option_State["uploaded_file"]["name"],
                "model_used": Option_State["plant_type"],
                "predictions": predictions,
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
        predictions, time_elapsed = run_on_image(image, n_stoma)
        record_predictions(
            predictions,
            filename,
            image.shape,
        )

        progress += increment
        progress_bar.progress(int(progress))

        with status_container:
            st.info(f"{filename} completed in {time_elapsed:.2f}s")

        total_time += time_elapsed
        n_stoma += predictions.n_predictions

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


def record_predictions(
    predictions,
    filename,
    image_size,
):
    path = "./output/temp/"
    store_population_filtering_measurements(predictions)
    filename = remove_extension_from_filename(filename)
    to_save = {
        "detections": predictions.detections,
        "image_size": image_size[:-1],
    }
    filepath = path + ".".join([filename, "json"])
    utils.write_to_json(to_save, filepath)


def store_population_filtering_measurements(predictions):
    Bounding_Boxes.extend(predictions.bounding_box_dimensions)
    Predicted_Pore_Lengths.extend(predictions.pore_lengths)


def remove_extension_from_filename(filename):
    file_extension = "." + filename.split(".")[-1]
    return filename.replace(file_extension, "")


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
        prediction["pore_width"] = convert_to_SIU_length(prediction["pore_width"])
        prediction["pore_length"] = convert_to_SIU_length(prediction["pore_length"])
        prediction["pore_area"] = convert_to_SIU_length(prediction["pore_area"])
        prediction["guard_cell_area"] = convert_to_SIU_length(
            prediction["guard_cell_area"]
        )
        prediction["subsidiary_cell_area"] = convert_to_SIU_length(
            prediction["subsidiary_cell_area"]
        )
    return predictions


def create_output_csvs():
    predictions = Option_State["folder_inference"]["predictions"]
    formatted_predictions = format_predictions(predictions)
    measurement_csv_filename = write_measurements_to_csv(formatted_predictions)
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
        for detection in detections:
            measurements = {
                "stoma_id": detection["stoma_id"],
                "pore_width": detection["pore_width"],
                "pore_length": detection["pore_length"],
                "pore_area": detection["pore_area"],
                "subsidiary_cell_area": detection["subsidiary_cell_area"],
                "guard_cell_area": detection["guard_cell_area"],
                "class": "open" if detection["category_id"] else "closed",
                "confidence": detection["confidence"],
                "image_name": image_name,
                "pore_width_to_length_ratio": detection["width_over_length"],
            }
            stoma_measurements.append(measurements)
    return stoma_measurements


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
    # Draw onto axis
    draw_measurements(ax, predictions)
    draw_bounding_boxes(ax, predictions)
    # Save drawing
    fig.savefig(
        os.path.join(output_path, image_name),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
