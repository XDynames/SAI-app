import os

import streamlit as st

from interface.upload_single import (
    setup_upload_sidebar,
    print_unavilable_message,
)
from tools.constants import IS_ONLINE
from tools.state import Option_State

def display_upload_zip():
    if IS_ONLINE:
        print_unavilable_message()
    else:
        image_folder_text_box()
        setup_upload_sidebar()

def image_folder_text_box():
    image_folder_path = st.text_input(
        "Enter the path to the folder containing your sample images:",
        value = Option_State['folder_path']
    )
    if os.path.exists(image_folder_path):
        Option_State['folder_path'] = image_folder_path
    else:
        st.warning('Please enter a valid directory path')
    