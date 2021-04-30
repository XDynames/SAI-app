import streamlit as st
import time

from app import utils
from inference.infer import run_on_image


def maybe_do_inference():
    if utils.is_file_uploaded():
        with st.spinner("Measuring Stoma..."):
            time.sleep(5)
            run_on_image()
        st.success("Finished")