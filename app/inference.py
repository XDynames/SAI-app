import streamlit as st
import time

from app import utils


def maybe_do_inference():
    if utils.is_file_uploaded():
        with st.spinner("Measuring Stoma..."):
            time.sleep(5)
        st.success("Finished")