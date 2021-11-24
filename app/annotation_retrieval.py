import os

from tools.load import download_json
from tools.state import Option_State
from app import utils


def get_predictions():
    if is_predictions_json_local():
        predictions = load_predictions()
    else:
        predictions = download_predictions()
    predictions = predictions["detections"]
    return predictions


def is_predictions_json_local():
    filepath = get_local_predictions_path()
    return os.path.exists(filepath)


def load_predictions():
    filepath = get_local_predictions_path()
    return utils.load_json(filepath)


def get_local_predictions_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + ".json")


def download_predictions():
    urls = utils.get_current_image_urls()
    return download_json(urls["predictions"] + "/download")


def get_ground_truth():
    if is_ground_truth_json_local():
        ground_truth = load_ground_truth()
    else:
        ground_truth = download_ground_truth()
    return ground_truth["detections"]


def is_ground_truth_json_local():
    filepath = get_local_ground_truth_path()
    return os.path.exists(filepath)


def get_local_ground_truth_path():
    species = Option_State["plant_type"].lower()
    image_name = Option_State["image_name"]
    return os.path.join(".", "assets", species, image_name + "-gt.json")


def load_ground_truth():
    filepath = get_local_ground_truth_path()
    return utils.load_json(filepath)


def download_ground_truth():
    urls = utils.get_current_image_urls()
    return download_json(urls["ground_truth"] + "/download")
