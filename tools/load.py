import os
import json
import urllib
from concurrent import futures

import cv2
import numpy as np
import streamlit as st

from .cloud_files import IMAGE_DICTS


@st.cache(show_spinner=True)
def download_image(url):
    with urllib.request.urlopen(url) as response:
        image = read_byte_stream(response)
    image = preprocess_image(image)
    return image


@st.cache(show_spinner=True)
def download_json(url):
    with urllib.request.urlopen(url) as response:
        downloaded_json = json.loads(response.read())
    return downloaded_json


def read_byte_stream(bytestream):
    return np.asarray(bytearray(bytestream.read()), dtype="uint8")


def preprocess_image(image):
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]
    return image


def download_assets(asset_url_dict, species, image_name):
    image_url = asset_url_dict["image"] + "/download"
    prediction_url = asset_url_dict["predictions"] + "/download"
    gt_url = asset_url_dict["ground_truth"] + "/download"
    image = download_image(image_url)
    image = preprocess_image(image)
    gt_json = download_json(gt_url)
    prediction_json = download_json(prediction_url)
    save_json(gt_json, species, image_name, gt=True)
    save_json(prediction_json, species, image_name)
    save_image(image, species, image_name)


def save_json(json_file, folder, filename, gt=False):
    filename += "-gt" if gt else ""
    filename += ".json"
    path = os.path.join(".", "assets", folder, filename)
    with open(path, "w") as file:
        json.dump(json_file, file)


def save_image(image, folder, filename):
    filename += ".png"
    path = os.path.join(".", "assets", folder, filename)
    cv2.imwrite(path, image)


def load_assets(cloud_files):
    for species in cloud_files.keys():
        for image_name in cloud_files[species].keys():
            image_dict = cloud_files[species][image_name]
            download_assets(image_dict, species, image_name)


# Current has some issues with streamlit
def async_load_assets(cloud_files):
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        for species in cloud_files.keys():
            for image_name in cloud_files[species].keys():
                image_dict = cloud_files[species][image_name]
                executor.submit(
                    download_assets,
                    image_dict,
                    species,
                    image_name
                )


def main():
    if not os.path.exists("./assets/"):
        os.mkdir("./assets/")
        os.mkdir("./assets/arabidopsis/")
        os.mkdir("./assets/barley/")
    async_load_assets(IMAGE_DICTS)


if __name__ == "__main__":
    main()
