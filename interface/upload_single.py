import streamlit as st

from interface.example_images import (
    plant_type_selection,
    confidence_sliderbar,
    immature_stomata_threshold,
)
from tools.load import decode_downloaded_image
from tools.state import Option_State
from tools.constants import (
    IS_ONLINE,
    OPENCV_FILE_SUPPORT,
)


def display_upload_image():
    if IS_ONLINE:
        print_unavilable_message()
    else:
        single_image_uploader()
        setup_upload_sidebar()


def print_unavilable_message():
    message = (
        "This feature is disabled in the online version of"
        " the application. To use this functionality please"
        " go to <INSERT LINK> to install and run the application"
        " locally."
    )
    st.write(message)


def single_image_uploader():
    file_like_object = st.file_uploader(
        "Upload Files", type=OPENCV_FILE_SUPPORT,
    )
    if file_like_object is not None:
        Option_State["uploaded_file"] = {
            'image': decode_downloaded_image(file_like_object),
            'name': file_like_object.name,
            }


def setup_upload_sidebar():
    plant_type_selection()
    confidence_sliderbar()
    camera_calibration_textbox()
    immature_stomata_threshold()


def camera_calibration_textbox():
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
