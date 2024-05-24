import os
import time

import torch
import detectron2
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

from inference.predictions import ModelOutput
from tools.cloud_files import EXTERNAL_DEPENDANCIES
from tools.load import download_and_save_yaml, download_and_save_model_weights
from tools.state import Option_State

BASE_CONFIDENCE_THRESHOLD = 0.6
Inference_Engines = {"Barley": None, "Arabidopsis": None}


class InferenceEngine:
    def __init__(self, model_config):
        self.metadata = MetadataCatalog.get(model_config.DATASETS.TEST[0])
        self.instance_mode = ColorMode.IMAGE
        self.predictor = DefaultPredictor(model_config)

    def run_on_image(self, image):
        with torch.no_grad():
            predictions = self.predictor(image)
        return predictions["instances"].to(torch.device("cpu"))


def run_on_image(image, n_stoma: int = 0):
    maybe_setup_inference_engine()
    start_time = time.time()
    demo = Inference_Engines[Option_State["plant_type"]]
    predictions = demo.run_on_image(image)
    time_elapsed = time.time() - start_time
    return ModelOutput(predictions, n_stoma), time_elapsed


def maybe_setup_inference_engine():
    selected_species = Option_State["plant_type"]
    if Inference_Engines[selected_species] is None:
        setup_inference_engine(selected_species)


def setup_inference_engine(selected_species):
    maybe_download_config_files(selected_species)
    maybe_download_model_weights(selected_species)
    configuration = setup_model_configuration(selected_species.lower())
    Inference_Engines[selected_species] = InferenceEngine(configuration)


def maybe_download_config_files(selected_species):
    selected_species = selected_species.lower()
    if not os.path.exists(f"./assets/config/{selected_species}.yaml"):
        filename = get_configuration_filepath(selected_species)
        url = EXTERNAL_DEPENDANCIES[f"base_{selected_species}_config"]
        download_and_save_yaml(url, filename)

    if not os.path.exists(f"./assets/config/{selected_species}_v2.yaml"):
        filename = get_configuration_filepath(f"{selected_species}_v2")
        url = EXTERNAL_DEPENDANCIES[f"{selected_species}_config"]
        download_and_save_yaml(url, filename)

    if not os.path.exists("./assets/config/Base-RCNN-FPN.yaml"):
        filename = get_configuration_filepath("Base-RCNN-FPN")
        url = EXTERNAL_DEPENDANCIES["base_config"]
        download_and_save_yaml(url, filename)


def maybe_download_model_weights(selected_species):
    selected_species = selected_species.lower()
    filepath = f"./assets/{selected_species}/weights.pth"
    url = EXTERNAL_DEPENDANCIES[f"{selected_species}_weights"]
    if not os.path.exists(filepath):
        download_and_save_model_weights(url, filepath)


def get_configuration_filepath(filename: str) -> str:
    return f"./assets/config/{filename}.yaml"


def get_model_weights_filepath(selected_species):
    return f"./assets/{selected_species}/weights.pth"


def setup_model_configuration(selected_species):
    cfg = detectron2.config.get_cfg()
    # a dirty fix for the keypoint resolution config
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = (14, 14)
    cfg.merge_from_file(get_configuration_filepath(f"{selected_species}_v2"))
    cfg.MODEL.WEIGHTS = get_model_weights_filepath(selected_species)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = BASE_CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = BASE_CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        BASE_CONFIDENCE_THRESHOLD
    )
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg
