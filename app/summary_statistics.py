import streamlit as st

from app import utils
from .annotation_retrieval import get_ground_truth, get_predictions
from .example_images import filter_low_confidence_predictions
from tools.constants import IS_ONLINE
from tools.state import Option_State


def maybe_display_summary_statistics():
    if utils.is_mode_view_examples():
        display_summary_statistics()
    if utils.is_mode_upload_an_example() and not IS_ONLINE:
        display_summary_statistics()


def display_summary_statistics():
    column_names, column_human, column_predicted = st.beta_columns(3)
    with column_names:
        display_summary_names()
    if utils.is_mode_upload_an_example() and Option_State['uploaded_inference'] is not None:
        with column_predicted:
            predictions = Option_State['uploaded_inference']['predictions']
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


def display_ground_truth_summary_statistics():
    ground_truth = get_ground_truth()
    st.write("Human Annotations")
    calculate_and_display_summary_statistics(ground_truth)


def display_prediction_summary_statistics(predictions=None):
    predictions = get_predictions() if predictions is None else predictions
    predictions = filter_low_confidence_predictions(predictions)
    st.write("Model Estimates")
    calculate_and_display_summary_statistics(predictions)


def calculate_and_display_summary_statistics(annotations):
    display_pore_count(annotations)
    display_average_length(annotations)
    display_average_width(annotations)
    display_average_area(annotations)


def display_pore_count(annotations):
    st.write(f"{len(annotations)}")


def display_pore_density(annotations):
    if is_valid_image_area():
        area = Option_State["image_area"]
        density = round(len(annotations) / area, 2)
        st.write(f"{density} stomata/mm\u00B2")
    else:
        st.write("N/A")


def is_valid_image_area():
    is_valid = False
    has_input = Option_State["uploaded_file"] is not None
    if has_input:
        is_valid = Option_State["image_area"] > 0.0001
    return is_valid


def display_average_length(annotations):
    average_length = average_key(annotations, "length")
    print_summary_metric(average_length, "\u03BCm", "px")


def display_average_width(annotations):
    average_width = average_key(annotations, "width")
    print_summary_metric(average_width, "\u03BCm", "px")


def display_average_area(annotations):
    average_area = average_key(annotations, "area")
    print_summary_metric(average_area, "\u03BCm\u00B2", "px\u00B2")


def average_key(annotations, key):
    values = [annotation[key] for annotation in annotations]
    if len(values) > 0:
        average = sum(values) / len(values)
    else:
        average = 0
    if is_valid_calibration():
        average /= Option_State["camera_calibration"]
    return round(average, 2)


def print_summary_metric(value, unit, default_unit):
    if is_valid_calibration():
        st.write(f"{value} {unit}")
    else:
        st.write(f"{value} {default_unit}")


def is_valid_calibration():
    return Option_State["camera_calibration"] > 0.0001
