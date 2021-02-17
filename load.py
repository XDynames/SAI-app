import json
import urllib
from concurrent import futures

import numpy as np
import streamlit as st


@st.cache(show_spinner=True)
def download_image(url):
    with urllib.request.urlopen(url) as response:
        image = read_byte_stream(response)
    return image


@st.cache(show_spinner=True)
def download_json(url):
    with urllib.request.urlopen(url) as response:
        downloaded_json = json.loads(response.read())
    return downloaded_json


def read_byte_stream(bytestream):
    return np.asarray(bytearray(bytestream.read()), dtype="uint8")


def download_assets(asset_url_dict):
    image_url = asset_url_dict['image'] + '/download'
    prediction_url = asset_url_dict['predictions'] + '/download'
    gt_url = asset_url_dict['ground_truth'] + '/download'
    download_image(image_url)
    download_json(gt_url)
    download_json(prediction_url)


def load_assets(cloud_files):
    for species in cloud_files.keys():
        for image_name in cloud_files[species].keys():
            image_dict = cloud_files[species][image_name]
            download_assets(image_dict)


# Current has some issues with streamlit
def async_load_assets(cloud_files):
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        for species in cloud_files.keys():
            for image_name in cloud_files[species].keys():
                image_dict = cloud_files[species][image_name]
                executor.submit(download_assets, image_dict)
