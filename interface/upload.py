import streamlit as st

from tools.state import Option_State
from tools.constants import (
    IS_ONLINE,
    OPENCV_FILE_SUPPORT,
)


def display_upload_image():
    if IS_ONLINE:
        message = (
            "This feature is disabled in the online version of"
            " the application. To use this functionality please"
            " go to <INSERT LINK> to install and run the application"
            " locally."
        )
        st.write(message)
    else:
        file_upload()
        draw_calibration_textboxes()


def file_upload():
    Option_State["uploaded_file"] = st.file_uploader(
        "Upload Files", type=OPENCV_FILE_SUPPORT
    )


def draw_calibration_textboxes():
    draw_camera_calibration_textbox()


def draw_camera_calibration_textbox():
    Option_State["camera_calibration"] = st.sidebar.number_input(
        "Camera Calibration (px/\u03BCm):",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.3f",
    )
    if Option_State["image_size"] is not None:
        image_size = Option_State["image_size"]
        area = convert_to_SIU_length(image_size[0]) * convert_to_SIU_length(
            image_size[1]
        )
        Option_State["image_area"] = area


def convert_to_SIU_length(pixel_length):
    return pixel_length * Option_State["camera_calibration"]
