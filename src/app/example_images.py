import copy

import matplotlib.pyplot as plt
import streamlit as st

from app import utils
from app.annotation_retrieval import get_predictions
from app.image_retrieval import get_selected_image
from inference.utils import convert_measurements
from tools import draw
from tools.constants import CAMERA_CALIBRATION
from tools import ground_truth
from tools.state import Option_State


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
        draw.legend(ax)
    else:
        draw.legend(ax)


def draw_predictions(mpl_axis, predictions=None):
    if predictions is None:
        predictions = get_predictions()
        detections = convert_measurements(predictions["detections"])
        maybe_draw_bboxes_on_image(mpl_axis, detections, False)
    invalid_detections = predictions["invalid_detections"]
    detections = apply_user_filters_to_predictions(detections)
    maybe_draw_labels_on_image(mpl_axis, detections, False)
    maybe_draw_bboxes_on_image(mpl_axis, invalid_detections, False)


def apply_user_filters_to_predictions(predictions):
    predictions = filter_low_confidence_predictions(predictions)
    predictions = filter_immature_stomata(predictions)
    return predictions


def draw_measurements(mpl_axis, predictions):
    predictions = apply_user_filters_to_predictions(predictions)
    draw.masks(mpl_axis, predictions, False)
    draw.keypoints(mpl_axis, predictions, False)


def draw_bounding_boxes(mpl_axis, predictions):
    draw.bboxes(mpl_axis, predictions, False)


def maybe_draw_predictions(mpl_axis):
    if Option_State["uploaded_inference"] is not None:
        predictions = Option_State["uploaded_inference"]["predictions"]
        invalid_predictions = Option_State["uploaded_inference"]["invalid_predictions"]
        predictions = convert_measurements(copy.deepcopy(predictions))
        draw_measurements(mpl_axis, predictions)
        draw_bounding_boxes(mpl_axis, predictions)
        draw.bboxes(mpl_axis, invalid_predictions, False)


def filter_low_confidence_predictions(predictions):
    threshold = Option_State["confidence_threshold"]
    return filter_predictions_below_threshold(predictions, threshold, "confidence")


def filter_immature_stomata(predictions):
    threshold = Option_State["minimum_stoma_length"]
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


def maybe_draw_labels_on_image(mpl_axis, annotations, gt):
    if Option_State["draw_masks"]:
        draw.masks(mpl_axis, annotations, gt)
    if Option_State["draw_keypoints"]:
        draw.keypoints(mpl_axis, annotations, gt)


def maybe_draw_bboxes_on_image(mpl_axis, annotations, gt):
    if Option_State["draw_bboxes"]:
        draw.bboxes(mpl_axis, annotations, gt)


def draw_ground_truth(mpl_axis):
    annotations = ground_truth.retrieve()
    maybe_draw_bboxes_on_image(mpl_axis, annotations, True)
    annotations = filter_immature_stomata(annotations)
    maybe_draw_labels_on_image(mpl_axis, annotations, True)
