import os

import streamlit as st

from interface.upload_single import convert_to_SIU_length
from tools.constants import (
    DENSITY_KEYS,
    DENSITY_OUTPUT_COLUMNS,
    MEASUREMENT_KEYS,
    MEASUREMENT_OUTPUT_COLUMN_NAMES,
)
from tools.state import Option_State


def create_output_csvs():
    predictions = Option_State["folder_inference"]["predictions"]
    formatted_predictions = format_predictions(predictions)
    measurement_csv_filename = write_measurements_to_csv(formatted_predictions)
    densities = format_densities(predictions)
    density_csv_filename = write_density_csv(densities)
    return density_csv_filename, measurement_csv_filename


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


def format_densities(predictions):
    densities = []
    for prediction in predictions:
        detections = prediction["detections"]
        invalid_detections = prediction["invalid_detections"]
        area = calculate_image_area(prediction["image_size"])
        n_stomata = len(detections) + len(invalid_detections)
        density = {
            "image_name": prediction["image_name"],
            "n_stomata": n_stomata,
            "density": n_stomata / area,
        }
        densities.append(density)
    return densities


def write_density_csv(densities):
    density_csv_filepath = write_output_csv(
        densities,
        DENSITY_OUTPUT_COLUMNS,
        DENSITY_KEYS,
        "density",
    )
    return density_csv_filepath


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


def calculate_image_area(image_size):
    height = convert_to_SIU_length(image_size[0])
    width = convert_to_SIU_length(image_size[1])
    if Option_State["camera_calibration"] > 0:
        height, width = height / 1000, width / 1000
    return height * width
