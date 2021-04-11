import os
import json

import cv2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from interface import ui
from tools.state import Option_State
from tools import draw
from tools.constants import IS_ONLINE
from tools.cloud_files import IMAGE_DICTS
from tools.load import (
    download_image,
    download_json,
    read_byte_stream,
)

is_setup = False


def main():
    maybe_setup()
    maybe_draw_example()
    maybe_display_summary_statistics()
    maybe_show_slide_output_example()


def maybe_setup():
    global is_setup
    if not is_setup:
        ui.setup()
        is_setup = True


def get_selected_image():
    if Option_State["mode"] == "View Example Images":
        image = get_example()
    if Option_State["mode"] == "Upload An Image":
        image = get_uploaded_image()
    return image


def get_uploaded_image():
    return read_byte_stream(Option_State["uploaded_file"])


def get_local_image_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + ".png")


def is_image_local():
    filepath = get_local_image_path()
    return os.path.exists(filepath)


def load_image():
    filepath = get_local_image_path()
    return cv2.imread(filepath)


def get_example():
    if is_image_local():
        image = load_image()
    else:
        image = download_example()
    return image


def download_example():
    urls = get_current_image_urls()
    return download_image(urls["image"] + "/download")


def get_current_image_urls():
    image_name = Option_State["image_name"]
    return Option_State["image_url_dicts"][image_name]


def download_annotations():
    urls = get_current_image_urls()
    return download_json(urls["annotations"] + "/download")


def download_predictions():
    urls = get_current_image_urls()
    return download_json(urls["predictions"] + "/download")


def download_ground_truth():
    urls = get_current_image_urls()
    return download_json(urls["ground_truth"] + "/download")


def draw_predictions(mpl_axis):
    predictions = get_and_filter_predictions()
    draw_labels_on_image(mpl_axis, predictions, False)


def is_predictions_json_local():
    filepath = get_local_predictions_path()
    return os.path.exists(filepath)


def load_predictions():
    filepath = get_local_predictions_path()
    return load_json(filepath)


def get_and_filter_predictions():
    if is_predictions_json_local():
        predictions = load_predictions()
    else:
        predictions = download_predictions()
    predictions = predictions["detections"]
    return filter_low_confidence_predictions(predictions)


def get_local_ground_truth_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + "-gt.json")


def get_local_predictions_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + ".json")


def is_ground_truth_json_local():
    filepath = get_local_ground_truth_path()
    return os.path.exists(filepath)


def load_ground_truth():
    filepath = get_local_ground_truth_path()
    return load_json(filepath)


def load_json(filepath):
    with open(filepath, "r") as file:
        json_dict = json.load(file)
    return json_dict


def get_ground_truth():
    if is_ground_truth_json_local():
        ground_truth = load_ground_truth()
    else:
        ground_truth = download_ground_truth()
    return ground_truth["detections"]


def filter_low_confidence_predictions(predictions):
    threshold = Option_State["confidence_threshold"]
    predictions = [
        prediction
        for prediction in predictions
        if prediction["confidence"] >= threshold
    ]
    return predictions


def draw_ground_truth(mpl_axis):
    ground_truth = get_ground_truth()
    draw_labels_on_image(mpl_axis, ground_truth, True)


def draw_labels_on_image(mpl_axis, annotations, gt):
    if Option_State["draw_bboxes"]:
        draw.bboxes(mpl_axis, annotations, gt)
    if Option_State["draw_masks"]:
        draw.masks(mpl_axis, annotations, gt)
    if Option_State["draw_keypoints"]:
        draw.keypoints(mpl_axis, annotations, gt)


def maybe_draw_example():
    if do_draw():
        draw_example()


def do_draw():
    draw_check_1 = is_drawing_mode() and is_file_uploaded()
    draw_check_2 = is_mode_view_examples()
    return draw_check_1 or draw_check_2


def is_drawing_mode():
    return not (is_mode_instructions() or is_mode_slide_output_example())


def is_mode_instructions():
    return Option_State["mode"] == "Instructions"


def is_file_uploaded():
    return Option_State["uploaded_file"] is not None


def is_mode_view_examples():
    return Option_State["mode"] == "View Example Images"


def is_mode_upload_an_example():
    return Option_State["mode"] == "Upload An Image"


def is_mode_slide_output_example():
    return Option_State["mode"] == "View Example Slide Output"


def maybe_display_summary_statistics():
    if is_mode_view_examples():
        display_summary_statistics()
    if is_mode_upload_an_example() and not IS_ONLINE:
        display_summary_statistics()


def display_summary_statistics():
    column_names, column_human, column_predicted = st.beta_columns(3)
    with column_names:
        display_summary_names()
    if not is_mode_upload_an_example():
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


def display_prediction_summary_statistics():
    predictions = get_and_filter_predictions()
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


def display_average_length(annotations):
    average_length = average_key(annotations, "length")
    print_summary_metric(average_length, "\u03BCm", "px")


def display_average_width(annotations):
    average_width = average_key(annotations, "width")
    print_summary_metric(average_width, "\u03BCm", "px")


def display_average_area(annotations):
    average_area = average_key(annotations, "area")
    print_summary_metric(average_area, "\u03BCm\u00B2", "px\u00B2")


def print_summary_metric(value, unit, default_unit):
    if is_valid_calibration():
        st.write(f"{value} {unit}")
    else:
        st.write(f"{value} {default_unit}")


def average_key(annotations, key):
    values = [annotation[key] for annotation in annotations]
    average = sum(values) / len(values)
    if is_valid_calibration():
        average /= Option_State["camera_calibration"]
    return round(average, 2)


def is_valid_calibration():
    return Option_State["camera_calibration"] > 0.0001


def is_valid_image_area():
    is_valid = False
    has_input = Option_State["uploaded_file"] is not None
    if has_input:
        is_valid = Option_State["image_area"] > 0.0001
    return is_valid


def maybe_show_slide_output_example():
    if is_mode_slide_output_example():
        url = IMAGE_DICTS["Barley"]["10Dec 19"]["predictions"] + "/download"
        predictions = download_json(url)
        df = pd.DataFrame(predictions["detections"])
        df = df.drop(columns=["bbox", "AB_keypoints", "CD_keypoints", "segmentation"])
        st.table(df.head(21))


def draw_example():
    image = get_selected_image()
    update_image_size(image)
    fig, ax = setup_plot(image)
    draw_annotations_on_image(ax)
    st.write(fig)


def draw_annotations_on_image(ax):
    if Option_State["draw_predictions"] and is_mode_view_examples():
        draw_predictions(ax)
    if Option_State["draw_ground_truth"] and is_mode_view_examples():
        draw_ground_truth(ax)
    # Add Resize here
    if is_mode_upload_an_example():
        draw.legend(ax, False)
    else:
        draw.legend(ax)


def update_image_size(image):
    Option_State["image_size"] = image.shape[:-1]


def setup_plot(image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    return fig, ax


if __name__ == "__main__":
    st.set_page_config(
        page_title="SAI - StomAI",
        page_icon=":purple_circle:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    main()
