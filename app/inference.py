import streamlit as st
import time

from app import utils
from app.image_retrieval import get_selected_image
from inference.infer import run_on_image
from app.example_images import setup_plot


def maybe_do_inference():
    if utils.is_file_uploaded() and utils.is_mode_upload_an_example():
        with st.spinner("Measuring Stoma..."):
            predictions, visualised = run_on_image(get_selected_image())
        st.success("Finished")
        st.write(visualised.fig)