import matplotlib.pyplot as plt
import streamlit as st

from tools import draw
from tools.state import Option_State
from app import utils
from .image_retrieval import get_selected_image
from .annotation_retrieval import get_ground_truth, get_predictions


def maybe_draw_example():
    if is_drawing():
        draw_example()


def is_drawing():
    draw_check_1 = utils.is_drawing_mode() and utils.is_file_uploaded()
    draw_check_2 = utils.is_mode_view_examples()
    return draw_check_1 or draw_check_2


def draw_example():
    image = get_selected_image()
    update_image_size(image)
    fig, ax = setup_plot(image)
    draw_annotations_on_image(ax)
    st.write(fig)


def update_image_size(image):
    Option_State["image_size"] = image.shape[:-1]


def setup_plot(image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    return fig, ax


def draw_annotations_on_image(ax):
    if Option_State["draw_predictions"] and utils.is_mode_view_examples():
        draw_predictions(ax)
    if Option_State["draw_ground_truth"] and utils.is_mode_view_examples():
        draw_ground_truth(ax)
    # Add Resize here
    if utils.is_mode_upload_an_example():
        draw.legend(ax, False)
    else:
        draw.legend(ax)


def draw_predictions(mpl_axis):
    predictions = get_predictions()
    predictions = filter_low_confidence_predictions(predictions)
    draw_labels_on_image(mpl_axis, predictions, False)


def filter_low_confidence_predictions(predictions):
    threshold = Option_State["confidence_threshold"]
    predictions = [
        prediction
        for prediction in predictions
        if prediction["confidence"] >= threshold
    ]
    return predictions


def draw_labels_on_image(mpl_axis, annotations, gt):
    if Option_State["draw_bboxes"]:
        draw.bboxes(mpl_axis, annotations, gt)
    if Option_State["draw_masks"]:
        draw.masks(mpl_axis, annotations, gt)
    if Option_State["draw_keypoints"]:
        draw.keypoints(mpl_axis, annotations, gt)


def draw_ground_truth(mpl_axis):
    ground_truth = get_ground_truth()
    draw_labels_on_image(mpl_axis, ground_truth, True)
