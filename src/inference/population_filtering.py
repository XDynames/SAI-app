import os

import streamlit as st
import numpy as np
from scipy.stats import iqr

from inference.utils import (
    calculate_bbox_height,
    calculate_bbox_width,
)
from app.utils import load_json, write_to_json

Predicted_Pore_Lengths, Bounding_Boxes = [], []


def remove_outliers_from_records():
    path = "./output/temp/"
    for filename in os.listdir(path):
        if ".json" in filename:
            if "-gt" in filename:
                continue
            filepath = os.path.join(path, filename)
            record = load_json(filepath)
            remove_outliers(record)
            write_to_json(record, filepath)


def remove_outliers(record):
    remove_pore_length_outliers(record)
    remove_bounding_box_outliers(record)


def remove_pore_length_outliers(record):
    indices, length_predictions = extract_pore_lengths(record)
    to_remove = find_outlier_indices(indices, length_predictions)
    remove_outlier_records(record, to_remove)


def find_outlier_indices(indices, lengths):
    to_remove = []
    lower_bound = calculate_length_limit()
    for i, length in enumerate(lengths):
        if length < lower_bound:
            to_remove.append(indices[i])
    return to_remove


def calculate_length_limit():
    inter_quartile_range = iqr(Predicted_Pore_Lengths, interpolation="midpoint")
    median = np.median(Predicted_Pore_Lengths)
    lower_whisker = median - 2.0 * inter_quartile_range
    return lower_whisker


def extract_pore_lengths(record):
    indices, lengths = [], []
    predictions = record["detections"]
    for i, prediction in enumerate(predictions):
        indices.append(i)
        lengths.append(prediction["pore_length"])
    return indices, lengths


def remove_bounding_box_outliers(record):
    remove_bounding_box_height_outliers(record)
    remove_bounding_box_width_outliers(record)


def remove_bounding_box_height_outliers(record):
    heights = extract_bbox_heights(record)
    minimum, maximum = calculate_bbox_height_limits()
    to_remove = find_outlier_bbox_indices(heights, minimum, maximum)
    remove_outlier_records(record, to_remove)


def remove_bounding_box_width_outliers(record):
    widths = extract_bbox_widths(record)
    minimum, maximum = calculate_bbox_width_limits()
    to_remove = find_outlier_bbox_indices(widths, minimum, maximum)
    remove_outlier_records(record, to_remove)


def extract_bbox_heights(record):
    heights = []
    predictions = record["detections"]
    for prediction in predictions:
        heights.append(calculate_bbox_height(prediction["bbox"]))
    return heights


def extract_bbox_widths(record):
    widths = []
    predictions = record["detections"]
    for prediction in predictions:
        widths.append(calculate_bbox_width(prediction["bbox"]))
    return widths


def find_outlier_bbox_indices(sizes, lower_bound, upper_bound):
    to_remove = set()
    for i, size in enumerate(sizes):
        if (size < lower_bound) or (size > upper_bound):
            to_remove.add(i)
    return to_remove


def calculate_bbox_height_limits():
    heights = [bbox["height"] for bbox in Bounding_Boxes]
    inter_quartile_range = iqr(heights, interpolation="midpoint")
    median = np.median(heights)
    lower_whisker = median - 2.0 * inter_quartile_range
    upper_whisker = median + 2.0 * inter_quartile_range
    return lower_whisker, upper_whisker


def calculate_bbox_width_limits():
    widths = [bbox["width"] for bbox in Bounding_Boxes]
    inter_quartile_range = iqr(widths, interpolation="midpoint")
    median = np.median(widths)
    lower_whisker = median - 2.0 * inter_quartile_range
    upper_whisker = median + 2.0 * inter_quartile_range
    return lower_whisker, upper_whisker


def remove_outlier_records(record, to_remove):
    predictions = record["detections"]
    removed = []
    for j in range(len(predictions))[::-1]:
        if j in to_remove:
            removed.append(predictions.pop(j))
    record["detections"] = predictions
    record["invalid_detections"].extend(removed)
