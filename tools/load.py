import json
import os
import pathlib
import requests
import urllib
import yaml
from concurrent import futures

import cv2
import numpy as np
import streamlit as st

from .cloud_files import IMAGE_DICTS
from .state import Option_State


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

def download_and_save_yaml(url, filename):
    save_yaml(filename, download_yaml(url))

def download_yaml(url):
    with urllib.request.urlopen(url) as response:
        downloaded_yaml = yaml.safe_load(response)
    return downloaded_yaml


def save_yaml(filename, downloaded_yaml):
    with open(filename, 'w') as file:
        yaml.dump(downloaded_yaml, file)


def download_and_save_model_weights(url, filename):
    filename = pathlib.Path(filename)
    response = requests.get(url)
    filename.write_bytes(response.content)


def decode_downloaded_image(bytestream):
    array = read_byte_stream(bytestream)
    return preprocess_image(array)


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
                executor.submit(download_assets, image_dict, species, image_name)


def main():
    maybe_create_folders()
    async_load_assets(IMAGE_DICTS)


def maybe_create_folders():
    if not os.path.exists("./assets/"):
        create_asset_folders()
    if not os.path.exists("./output/"):
        create_output_folders()


def create_asset_folders():
    os.mkdir("./assets/")
    os.mkdir("./assets/arabidopsis/")
    os.mkdir("./assets/barley/")
    os.mkdir("./assets/config/")


def create_output_folders():
    os.mkdir("./output/")
    os.mkdir("./output/temp/")


def clean_temporary_folder():
    path = './output/temp/'
    files = os.listdir(path)
    for file in files:
        os.remove(path + file)


def maybe_create_visualisation_folder():
    visualisation_path = Option_State['visualisation_path']
    if not os.path.exists(visualisation_path):
        os.makedirs(visualisation_path)


if __name__ == "__main__":
    main()
