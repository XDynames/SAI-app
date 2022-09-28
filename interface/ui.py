import base64

import streamlit as st

from tools.state import Option_State
from .landing import display_instructions
from .upload_single import display_upload_image
from .upload_folder import display_upload_zip
from .example_images import display_example_selection
from .example_output import (
    display_group_output_example,
    display_slide_output_example,
)

ENABLED_MODES = [
    "Instructions",
    "View Example Images",
    "View Example Slide Output",
    "Upload An Image",
    "Upload Multiple Images",
]

MODE_METHODS = {
    "Instructions": display_instructions,
    "Upload An Image": display_upload_image,
    "View Example Images": display_example_selection,
    "View Example Slide Output": display_slide_output_example,
    "View Example Group Output": display_group_output_example,
    "Upload Multiple Images": display_upload_zip,
}


def setup():
    draw_title()
    setup_sidebar()


def draw_title():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            text-align: center;
            font-weight:700 !important;
            font-size:35px !important;
            color: #00000 !important;
            padding-top: 15px !important;
            padding-left: 20px;
        }
        .logo-text subtext {
            font-weight:500 !important;
            font-size:25px !important;
            color: #00000 !important;
        }
        .logo-img {
            float:right;
            width: 168px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('logos/sai.png', "rb").read()).decode()}">
            <p class="logo-text">
            StomaAI <br style="line-height: 1px"/>
            <subtext>Accelerating Plant Physiology Research</subtext>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def setup_sidebar():
    mode_selection()
    MODE_METHODS[Option_State["mode"]]()


def mode_selection():
    Option_State["mode"] = st.sidebar.selectbox(
        "Select Application Mode:", ENABLED_MODES
    )
