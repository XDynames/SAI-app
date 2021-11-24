import pandas as pd
import streamlit as st

from app import utils
from tools.cloud_files import IMAGE_DICTS
from tools.load import download_json


def maybe_show_slide_output_example():
    if utils.is_mode_slide_output_example():
        url = "https://cloudstor.aarnet.edu.au/plus/s/DXBhmYRCZYKyk7R" + "/download"
        df = pd.read_csv(url)
        st.table(df.head(30))
