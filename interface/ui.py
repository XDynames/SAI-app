import streamlit as st

from tools.state import Option_State
from .landing import display_instructions
from .upload import display_upload_image
from .example_images import display_example_selection
from .example_output import display_group_output_example, display_slide_output_example

ENABLED_MODES = [
    "Instructions",
    "View Example Images",
    "View Example Slide Output",
    "Upload An Image",
]

MODE_METHODS = {
    "Instructions": display_instructions,
    "Upload An Image": display_upload_image,
    "View Example Images": display_example_selection,
    "View Example Slide Output": display_slide_output_example,
    "View Example Group Output": display_group_output_example,
}


def setup():
    draw_title()
    setup_sidebar()


def draw_title():
    heading = "<h1 style='text-align: center'>SAI - StomaAI</h1>"
    st.markdown(heading, unsafe_allow_html=True)


def setup_sidebar():
    mode_selection()
    MODE_METHODS[Option_State["mode"]]()


def mode_selection():
    Option_State["mode"] = st.sidebar.selectbox(
        "Select Application Mode:", ENABLED_MODES
    )
