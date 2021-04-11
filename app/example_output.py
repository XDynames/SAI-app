import pandas as pd
import streamlit as st

from app import utils
from tools.cloud_files import IMAGE_DICTS
from tools.load import download_json


def maybe_show_slide_output_example():
    if utils.is_mode_slide_output_example():
        url = IMAGE_DICTS["Barley"]["10Dec 19"]["predictions"] + "/download"
        predictions = download_json(url)
        df = pd.DataFrame(predictions["detections"])
        df = df.drop(columns=["bbox", "AB_keypoints", "CD_keypoints", "segmentation"])
        st.table(df.head(21))
