import pandas as pd
import streamlit as st

from app import utils
from tools.cloud_files import EXAMPLE_MEASUREMENT_OUTPUTS


def maybe_show_slide_output_example():
    if utils.is_mode_slide_output_example():
        url = EXAMPLE_MEASUREMENT_OUTPUTS["sample"] + "/download"
        df = pd.read_csv(url)
        st.table(df.head(10))
        st.write("### Population Measurement Output")
        message = (
            "Below is an Example of .csv file output for model"
            " predictions on a population of images."
            " In this example, each row contains measurements"
            " for a different image."
        )
        st.write(message)
        url = EXAMPLE_MEASUREMENT_OUTPUTS["population"] + "/download"
        df = pd.read_csv(url)
        st.table(df.head(5))
