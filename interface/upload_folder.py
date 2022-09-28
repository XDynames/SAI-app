import os

import streamlit as st

from interface.upload_single import (
    setup_upload_sidebar,
    print_unavilable_message,
)
from interface.folder_select import display_folder_selection
from tools.constants import IS_ONLINE, OPENCV_FILE_SUPPORT
from tools.state import Option_State
from .utils import download_button


def display_upload_zip():
    if IS_ONLINE:
        print_unavilable_message()
    else:
        if Option_State["select_folder"]:
            display_folder_selection()
        else:
            image_folder_text_box()
            setup_upload_sidebar()


def image_folder_text_box():
    columns = st.columns((3, 1))
    with columns[0]:
        image_folder_path = st.text_input(
            "Currently selected folder to measure images in:",
            value=Option_State["current_path"],
        )
    with columns[1]:
        st.button("Browse folders", on_click=set_mode_to_folder_selection)
        Option_State["infer_button"] = st.button("Measure!")

    if os.path.exists(image_folder_path):
        Option_State["folder_path"] = image_folder_path
    else:
        st.warning("Please enter a valid directory path")

    if not is_image_in_folder():
        st.warning("There are no images in the currently selected folder")


def set_mode_to_folder_selection():
    Option_State["select_folder"] = True


def is_image_in_folder():
    contents = os.listdir(Option_State["current_path"])
    is_image = [is_supported_image(content) for content in contents]
    return any(is_image)


def is_supported_image(filename):
    supported_files = set(OPENCV_FILE_SUPPORT)
    return filename.split(".")[-1] in supported_files


def show_download_csv_button(csv_download, csv_filename):
    csv_download_button = download_button(
        csv_download, csv_filename, "Download Measurements"
    )
    st.write(csv_download_button, unsafe_allow_html=True)


def show_save_visualisations_options():
    output_path = os.path.join(
        Option_State["current_path"], "visualised_measurements"
    )
    Option_State["visualisation_path"] = st.text_input(
        "Enter the path where you want to save the images:", value=output_path
    )
    Option_State["visualise"] = st.button("Save Measurement Visualisations")
