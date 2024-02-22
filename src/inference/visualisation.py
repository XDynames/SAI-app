import os

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from app.example_images import (
    draw_bounding_boxes,
    draw_measurements,
    setup_plot,
)
from app import utils
from inference.utils import get_list_of_images_in_folder
from tools.load import maybe_create_visualisation_folder
from tools.state import Option_State


def maybe_visualise_and_save():
    if Option_State["visualise"]:
        maybe_create_visualisation_folder()
        visualise_and_save()
        Option_State["visualise"] = False


def visualise_and_save():
    visualisation_folder = Option_State["visualisation_path"]
    image_folder = Option_State["folder_path"]
    image_names = get_list_of_images_in_folder(image_folder)

    status_container = st.empty()
    progress = 0
    progress_bar_header = st.empty()
    with progress_bar_header:
        st.write(f"Visualising {len(image_names)} images...")
    progress_bar = st.progress(progress)
    increment = 100 / len(image_names)

    for image_name in image_names:
        draw_and_save_visualisation(image_name)
        progress += increment
        progress_bar.progress(int(progress))

    progress_bar.progress(100)
    progress_bar.empty()
    progress_bar_header.empty()
    with status_container:
        st.success(
            f"{len(image_names)} visualised images are now saved in {visualisation_folder}"
        )


def draw_and_save_visualisation(image_name):
    output_path = Option_State["visualisation_path"]
    input_path = Option_State["folder_path"]
    # Load image
    image = cv2.imread(os.path.join(input_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create matplot lib axis
    fig, ax = setup_plot(image)
    # Load images measurements
    predictions_filename = image_name.split(".")[0] + ".json"
    prediction_path = os.path.join("./output/temp/", predictions_filename)
    record = utils.load_json(prediction_path)
    predictions = record["detections"]
    # Draw onto axis
    draw_measurements(ax, predictions)
    draw_bounding_boxes(ax, predictions)
    # Save drawing
    fig.savefig(
        os.path.join(output_path, image_name),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
