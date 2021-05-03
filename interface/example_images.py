import streamlit as st

from tools.cloud_files import IMAGE_DICTS
from tools.state import Option_State
from tools.constants import PLANT_OPTIONS, CAMERA_CALIBRATION


def display_example_selection():
    plant_type_selection()
    image_selection()
    immature_stomata_threshold()
    show_calibration_information()
    drawing_options()


def plant_type_selection():
    Option_State["plant_type"] = st.sidebar.selectbox(
        "Select a plant type:", PLANT_OPTIONS
    )


def image_selection():
    image_dict = IMAGE_DICTS[Option_State["plant_type"]]
    Option_State["image_url_dicts"] = image_dict
    Option_State["image_name"] = st.sidebar.selectbox(
        "Select image:", [image_name for image_name in image_dict.keys()]
    )


def immature_stomata_threshold():
    Option_State["minimum_stoma_length"] = st.sidebar.number_input(
        "Immature Stomata Threshold (\u03BCm):",
        min_value=0.0,
        value=0.0,
        step=10.0,
        format="%.3f",
    )


def show_calibration_information():
    print_camera_calibration()


def print_camera_calibration():
    camera_calibration = CAMERA_CALIBRATION[Option_State["plant_type"]]
    Option_State["camera_calibration"] = camera_calibration
    message = f"Camera Calibration: {camera_calibration:.4} px/\u03BCm"
    st.sidebar.write(message)


def drawing_options():
    draw_ground_truth_checkbox()
    draw_predictions_checkbox()
    maybe_draw_confidence_slidebar()
    maybe_draw_annotation_options()


def draw_ground_truth_checkbox():
    Option_State["draw_ground_truth"] = st.sidebar.checkbox("Show Human Measurements")


def draw_predictions_checkbox():
    Option_State["draw_predictions"] = st.sidebar.checkbox("Show Model Predictions")


def maybe_draw_confidence_slidebar():
    if draw_predictions_enabled():
        confidence_sliderbar()


def draw_predictions_enabled():
    return Option_State["draw_predictions"]


def confidence_sliderbar():
    Option_State["confidence_threshold"] = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


def maybe_draw_annotation_options():
    if drawing_enabled():
        st.sidebar.write("Types of Measurments:")
        draw_bboxes_checkbox()
        draw_masks_checkbox()
        draw_keypoints_checkbox()


def drawing_enabled():
    drawing_ground_truth = Option_State["draw_ground_truth"]
    return draw_predictions_enabled() or drawing_ground_truth


def draw_bboxes_checkbox():
    Option_State["draw_bboxes"] = st.sidebar.checkbox(
        "Show Bounding Boxes",
        value=True,
    )


def draw_masks_checkbox():
    Option_State["draw_masks"] = st.sidebar.checkbox(
        "Show Pore Segmentations",
        value=True,
    )


def draw_keypoints_checkbox():
    Option_State["draw_keypoints"] = st.sidebar.checkbox(
        "Show Lengths and Widths",
        value=True,
    )
