import matplotlib.pyplot as plt
import streamlit as st

from tools import draw
from tools.constants import CAMERA_CALIBRATION
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
    image = image[:, :, [2, 1, 0]]
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
    if utils.is_mode_upload_an_example():
        maybe_draw_predictions(ax)
        draw.legend(ax, False)
    else:
        draw.legend(ax)


def draw_predictions(mpl_axis, predictions=None):
    predictions = get_predictions() if predictions is None else predictions
    predictions = apply_user_filters_to_predictions(predictions)
    draw_labels_on_image(mpl_axis, predictions, False)


def apply_user_filters_to_predictions(predictions):
    predictions = filter_low_confidence_predictions(predictions)
    predictions = filter_immature_stomata(predictions)
    return predictions


def draw_measurements(mpl_axis, predictions):
    predictions = apply_user_filters_to_predictions(predictions)
    draw.masks(mpl_axis, predictions, False)
    draw.keypoints(mpl_axis, predictions, False)


def draw_bounding_boxes(mpl_axis, predictions):
    predictions = apply_user_filters_to_predictions(predictions)
    draw.bboxes(mpl_axis, predictions, False)


def maybe_draw_predictions(mpl_axis):
    if Option_State["uploaded_inference"] is not None:
        predictions = Option_State["uploaded_inference"]["predictions"]
        valid_indices = Option_State["uploaded_inference"]["valid_detection_indices"]
        valid_predictions = [predictions[i] for i in valid_indices]
        draw_measurements(mpl_axis, valid_predictions)
        draw_bounding_boxes(mpl_axis, predictions)


def filter_low_confidence_predictions(predictions):
    threshold = Option_State["confidence_threshold"]
    return filter_predictions_below_threshold(predictions, threshold, "confidence")


def filter_immature_stomata(predictions):
    threshold_in_micron = Option_State["minimum_stoma_length"]
    threshold = threshold_in_micron * get_pixel_to_micron_conversion_factor()
    return filter_predictions_below_threshold(predictions, threshold, "pore_length")


def get_pixel_to_micron_conversion_factor():
    if utils.is_mode_view_examples():
        pixels_per_micron = CAMERA_CALIBRATION[Option_State["plant_type"]]
    else:
        pixels_per_micron = Option_State["camera_calibration"]
    return pixels_per_micron


def filter_predictions_below_threshold(predictions, threshold, key):
    predictions = [
        prediction for prediction in predictions if prediction[key] >= threshold
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
