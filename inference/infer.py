import os
import time
import torch
import streamlit as st

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from tools.cloud_files import EXTERNAL_DEPENDANCIES
from tools.load import download_and_save_yaml, download_and_save_model_weights
from tools.state import Option_State

Inference_Engines = {'Barley': None, 'Arabidopsis': None}

class InferenceEngine:
    def __init__(self, model_config):
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.instance_mode = ColorMode.IMAGE
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(cfg)
    
    def run_on_image(self, image):
        predictions = self.predictor(image)
        self._post_process_predictions(predictions)
        vis_output = self._visualise_predictions(predictions, image)
        return predictions, vis_output
    
    def _post_process_predictions(self, predictions):
        instances = predictions["instances"].to(self.cpu_device)
        filter_invalid_predictions(instances)
        predictions['instances'] = instances
    
    def _visualise_predictions(self, predictions, image):
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualiser = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        self._patch_visualiser_so_it_draws_thin_lines(visualiser)
        return visualiser.draw_instance_predictions(predictions=instances)
    
    def _patch_visualiser_so_it_draws_thin_lines(self, visualiser):
        # Monkey Patch to draw thinner lines
        def draw_thin_line(self, x_data, y_data, color, linestyle="-", linewidth=2):
            self._draw_line(x_data, y_data, color, "-", linewidth)
        visualiser._draw_line = visualiser.draw_line
        visualiser.draw_line = types.MethodType(draw_thin_line, visualiser)


def run_on_image():
    maybe_setup_inference_engine()
    start_time = time.time()
    #predictions, visualised_output = demo.run_on_image(image)
    time_elapsed = time.time() - start_time


def maybe_setup_inference_engine():
    selected_species = Option_State["plant_type"]
    if Inference_Engines[selected_species] is None:
        setup_inference_engine()

def setup_inference_engine():
    selected_species = Option_State["plant_type"]
    st.write(f"Hey dude looks like you want to do some inference on: {selected_species}")

    maybe_download_config_files(selected_species)
    maybe_download_model_weights(selected_species)
    # Download dependencies
    #    config file
    #    weights -> asynch call for weights?

    # Instance gloabl InferenceEngine
    #   one for arabidopsis one for barley?

def maybe_download_config_files(selected_species):
    if not os.path.exists(f'./assets/config/{selected_species}.yaml'):
        filename = f'./assets/config/{selected_species}.yaml'
        url = EXTERNAL_DEPENDANCIES[f"{selected_species}_config"] + "/download"
        download_and_save_yaml(url, filename)
    
    if not os.path.exists('./assets/config/Base-RCNN-FPN.yaml'):
        filename = './assets/config/Base-RCNN-FPN.yaml'
        url = EXTERNAL_DEPENDANCIES["base_config"] + "/download"
        download_and_save_yaml(url, filename)

def maybe_download_model_weights(selected_species):
    filepath = f'./assets/{selected_species.lower()}/weights.pth'
    url = EXTERNAL_DEPENDANCIES[f'{selected_species}_weights'] + '/download'
    if not os.path.exists(filepath):
        download_and_save_model_weights(url, filepath)