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
        " go to this [link](https://github.com/XDynames/SAI-app)"
        " to install and run the application locally."
    )
    st.markdown(message)


def single_image_uploader():
    file_like_object = st.file_uploader(
        "Upload Files",
        type=OPENCV_FILE_SUPPORT,
    )
    if file_like_object is not None:
        Option_State["uploaded_file"] = {
            "image": decode_downloaded_image(file_like_object),
            "name": file_like_object.name,
        }


def setup_upload_sidebar():
    plant_type_selection()
    confidence_sliderbar()
    camera_calibration_textbox()
    immature_stomata_threshold()


def camera_calibration_textbox():
    Option_State["camera_calibration"] = st.sidebar.number_input(
        "Camera Calibration (px/\u03BCm):",
        min_value=0.000000,
        value=0.00000,
        step=0.00001,
        format="%.5f",
    )
    if Option_State["image_size"] is not None:
        image_size = Option_State["image_size"]
        height = convert_to_SIU_length(image_size[0])
        width = convert_to_SIU_length(image_size[1])
        if Option_State["camera_calibration"] > 0:
            height /= 1000
            width /= 1000
        Option_State["image_area"] = width * height


def convert_to_SIU_length(pixel_length):
    if Option_State["camera_calibration"] > 0:
        length = pixel_length / Option_State["camera_calibration"]
    else:
        length = pixel_length
    return length


def convert_to_SIU_area(pixel_area):
    if Option_State["camera_calibration"] > 0:
        area = pixel_area / Option_State["camera_calibration"] ** 2
    else:
        area = pixel_area
    return area
