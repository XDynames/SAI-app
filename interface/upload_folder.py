import os

import streamlit as st

from interface.upload_single import (
    setup_upload_sidebar,
    print_unavilable_message,
)
from tools.constants import IS_ONLINE
from tools.state import Option_State
from .utils import download_button


def display_upload_zip():
    if IS_ONLINE:
        print_unavilable_message()
    else:
        image_folder_text_box()
        setup_upload_sidebar()


def image_folder_text_box():
    image_folder_path = st.text_input(
        "Enter the path to the folder containing your sample images:",
        value=Option_State['folder_path']
    )
    if os.path.exists(image_folder_path):
        Option_State['folder_path'] = image_folder_path
    else:
        st.warning('Please enter a valid directory path')


def show_download_csv_button(csv_download, csv_filename):
    csv_download_button = download_button(
        csv_download,
        csv_filename,
        'Download Measurements'
    )
    st.write(csv_download_button, unsafe_allow_html=True)


def show_save_visualisations_options():
    if Option_State['visualisation_path'] is None:
        output_path = os.path.join(Option_State['folder_path'], "visualised_measurements")
    else:
        output_path = Option_State['visualisation_path']
    Option_State['visualisation_path'] = st.text_input(
        "Enter the path where you want to save the images:",
        value=output_path
    )
    Option_State['visualise'] = st.button("Save Measurement Visualisations")
