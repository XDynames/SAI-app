import streamlit as st
from typing import Dict, List, Union

from app import utils
from .annotation_retrieval import get_predictions
from .example_images import (
    filter_immature_stomata,
    filter_low_confidence_predictions,
)
from inference.output import calculate_g_max
from tools import ground_truth
from tools.constants import IS_ONLINE, IMAGE_AREA
from tools.state import Option_State


def maybe_display_summary_statistics():
    if utils.is_mode_view_examples():
        display_summary_statistics()
    if utils.is_mode_upload_an_example() and not IS_ONLINE:
        display_summary_statistics()


def display_summary_statistics():
    column_names, column_human, column_predicted = st.columns(3)
    with column_names:
        display_summary_names()
    if (
        utils.is_mode_upload_an_example()
        and Option_State["uploaded_inference"] is not None
    ):
        with column_predicted:
            predictions = Option_State["uploaded_inference"]["predictions"]
            display_prediction_summary_statistics(predictions)
    if not utils.is_mode_upload_an_example():
        with column_human:
            display_ground_truth_summary_statistics()
        with column_predicted:
            display_prediction_summary_statistics()


def display_summary_names():
    st.write("Properties")
    st.write("Stomata Count:")
    st.write("Average Pore Length:")
    st.write("Average Pore Width:")
    st.write("Average Pore Area:")
    st.write("Stomatal Density:")
    st.write("Estimated g max:")


def display_ground_truth_summary_statistics():
    annotations = ground_truth.retrieve()
    st.write("Human Annotations")
    display_pore_count(annotations)
    calculate_and_display_summary_statistics(annotations)


def display_prediction_summary_statistics(predictions=None):
    st.write("Model Estimates")
    if predictions is None:
        predictions = get_predictions()
        detections = predictions["detections"]
        invalid_detections = predictions["invalid_detections"]
        detections = apply_user_filters(detections)
        display_pore_count(detections, invalid_detections)
        calculate_and_display_summary_statistics(detections, invalid_detections)
    else:
        predictions = apply_user_filters(predictions)
        display_pore_count(predictions)
        calculate_and_display_summary_statistics(predictions)


def calculate_and_display_summary_statistics(
    annotations: List[Dict],
    invalid_detections: List[Dict] = [],
):
    display_average_length(annotations)
    display_average_width(annotations)
    display_average_area(annotations)
    display_pore_density(annotations, invalid_detections)
    display_g_max(annotations, invalid_detections)


def display_pore_count(detections: List[Dict], invalid_detections: List[Dict] = []):
    n_stomata = len(detections) + len(invalid_detections)
    st.write(f"{n_stomata}")


def apply_user_filters(predictions):
    predictions = filter_low_confidence_predictions(predictions)
    predictions = filter_immature_stomata(predictions)
    return predictions


def display_pore_density(
    annotations: List[Dict],
    invalid_detections: List[Dict],
):
    density = calculate_sample_density(annotations, invalid_detections)
    if density > 0.0:
        density = round(density, 2)
        st.write(f"{density} stomata/mm\u00B2")
    else:
        st.write("N/A")


def calculate_sample_density(
    annotations: List[Dict],
    invalid_detections: List[Dict],
) -> float:
    area = get_image_area()
    if area > 0.001:
        n_stomata = len(annotations) + len(invalid_detections)
        density = n_stomata / area
    else:
        density = 0.0
    return density


def get_image_area() -> float:
    area = 0
    if is_valid_image_area():
        area = Option_State["image_area"]
    elif utils.is_mode_view_examples():
        area = IMAGE_AREA[Option_State["plant_type"]]
    return area


def is_valid_image_area():
    is_valid = False
    has_input = Option_State["uploaded_file"] is not None
    if has_input:
        is_valid = Option_State["image_area"] > 0.0001
    return is_valid


def display_g_max(
    annotations: List[Dict],
    invalid_detections: List[Dict],
):
    g_max = calculate_sample_g_max(annotations, invalid_detections)
    if g_max > 0.0:
        g_max = round(g_max, 2)
        st.write(f"{g_max} mol/m\u00B2s")
    else:
        st.write("N/A")


def calculate_sample_g_max(
    annotations: List[Dict],
    invalid_detections: List[Dict],
) -> float:
    density = calculate_sample_density(annotations, invalid_detections)
    return calculate_g_max(density, annotations)


def display_average_length(annotations):
    average_length = average_key(annotations, "pore_length")
    print_summary_metric(average_length, "\u03BCm", "px")


def display_average_width(annotations):
    average_width = average_key(annotations, "pore_width")
    print_summary_metric(average_width, "\u03BCm", "px")


def display_average_area(annotations):
    average_area = average_key(annotations, "pore_area")
    print_summary_metric(average_area, "\u03BCm\u00B2", "px\u00B2")


def average_key(annotations, key):
    values = [annotation[key] for annotation in annotations]
    if len(values) > 0:
        average = sum(values) / len(values)
    else:
        average = 0
    if is_valid_calibration():
        average /= Option_State["camera_calibration"]
        if key == "area":
            average /= Option_State["camera_calibration"]
    return round(average, 2)


def print_summary_metric(value, unit, default_unit):
    if is_valid_calibration():
        st.write(f"{value} {unit}")
    else:
        st.write(f"{value} {default_unit}")


def is_valid_calibration():
    return Option_State["camera_calibration"] > 0.0001
