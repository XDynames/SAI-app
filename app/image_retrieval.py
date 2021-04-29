import os

import cv2

from tools.state import Option_State
from tools.load import download_image, read_byte_stream, preprocess_image
from app import utils


def get_selected_image():
    if Option_State["mode"] == "View Example Images":
        image = get_example()
    if Option_State["mode"] == "Upload An Image":
        image = get_uploaded_image()
    return image


def get_example():
    if is_image_local():
        image = load_image()
    else:
        image = download_example()
    return image


def is_image_local():
    filepath = get_local_image_path()
    return os.path.exists(filepath)


def load_image():
    filepath = get_local_image_path()
    return cv2.imread(filepath)


def get_local_image_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + ".png")


def download_example():
    urls = utils.get_current_image_urls()
    return download_image(urls["image"] + "/download")


def get_uploaded_image():
    return preprocess_image(read_byte_stream(Option_State["uploaded_file"]))
