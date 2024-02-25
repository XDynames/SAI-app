import os
import math
from typing import Dict, List, Tuple

from interface.upload_single import convert_to_SIU_length
from inference.constants import (
    DENSITY_KEYS,
    DENSITY_OUTPUT_COLUMNS,
    DIFFUSIVITY_OF_WATER_IN_AIR_25C,
    MEASUREMENT_KEYS,
    MEASUREMENT_OUTPUT_COLUMN_NAMES,
    MOLAR_VOLUME_OF_WATER_IN_AIR_25C,
)
from tools.state import Option_State


def create_output_csvs() -> Tuple[str, str]:
    predictions = Option_State["folder_inference"]["predictions"]
    formatted_predictions = format_predictions(predictions)
    measurement_csv_filename = write_measurements_to_csv(formatted_predictions)
    densities = format_densities(predictions)
    density_csv_filename = write_density_csv(densities)
    return density_csv_filename, measurement_csv_filename


def format_predictions(predictions: List[Dict]) -> List[Dict]:
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


def write_measurements_to_csv(measurements: List[Dict]) -> str:
    measurement_csv_filepath = write_output_csv(
        measurements,
        MEASUREMENT_OUTPUT_COLUMN_NAMES,
        MEASUREMENT_KEYS,
        "pore_measurements",
    )
    return measurement_csv_filepath


def format_densities(predictions: List[Dict]) -> List[Dict]:
    densities = []
    for prediction in predictions:
        detections = prediction["detections"]
        invalid_detections = prediction["invalid_detections"]
        area = calculate_image_area(prediction["image_size"])
        n_stomata = len(detections) + len(invalid_detections)
        density = n_stomata / area  # pores/mm^2
        g_max = calculate_g_max(density, prediction)
        density = {
            "image_name": prediction["image_name"],
            "n_stomata": n_stomata,
            "density": density,
            "g_max": g_max,
        }
        densities.append(density)
    return densities


def write_density_csv(densities: List[Dict]) -> str:
    density_csv_filepath = write_output_csv(
        densities,
        DENSITY_OUTPUT_COLUMNS,
        DENSITY_KEYS,
        "density",
    )
    return density_csv_filepath


def write_output_csv(
    measurements: List[Dict],
    column_names: List[str],
    column_keys: List[str],
    name: str,
) -> str:
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


def calculate_image_area(image_size: List[float]) -> float:
    height = convert_to_SIU_length(image_size[0])
    width = convert_to_SIU_length(image_size[1])
    if Option_State["camera_calibration"] > 0:
        # Convert to mm
        height, width = height / 1000, width / 1000
    return height * width


def calculate_g_max(density: float, prediction: Dict) -> float:
    pore_depth = caclulate_pore_depth(prediction)
    a_max = caclulate_a_max(prediction)
    constant = DIFFUSIVITY_OF_WATER_IN_AIR_25C / MOLAR_VOLUME_OF_WATER_IN_AIR_25C
    numerator = a_max * density
    denominator = pore_depth + math.sqrt(a_max * math.pi / 4)
    g_max_mmol = constant * numerator / denominator
    return g_max_mmol / 1000


def caclulate_pore_depth(prediction: Dict) -> float:
    width = calculate_average_length_of_key(prediction, "guard_cell_width")
    return width / 2


def caclulate_a_max(prediction: Dict) -> float:
    a_max = 0.0
    if Option_State["plant_type"] == "Barley":
        a_max = calculate_monocot_a_max(prediction)
    else:
        a_max = calculate_dicot_a_max(prediction)
    return a_max


def calculate_monocot_a_max(prediction: Dict) -> float:
    groove_length = calculate_average_length_of_key(
        prediction,
        "guard_cell_groove_length",
    )
    groove_length /= 2
    pore_length = calculate_average_length_of_key(prediction, "pore_length")
    pore_length /= 2
    return math.pi * groove_length * pore_length


def calculate_dicot_a_max(prediction: Dict) -> float:
    pore_length = calculate_average_length_of_key(prediction, "pore_length")
    pore_length /= 2
    return math.pi * pore_length**2


def calculate_average_length_of_key(prediction: Dict, key: str) -> float:
    lengths = [detection[key] for detection in prediction["detections"]]
    return sum(lengths) / len(lengths)
