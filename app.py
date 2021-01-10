import json
import urllib

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import ui
import draw
from ui import Option_State

# mm2
IMAGE_AREA = {
    'Barley' : 0.3229496,
    'Arabidopsis' : 0.04794822,
}
# pixels / micron
CAMERA_CALIBRATION = {
    'Barley' : 4.2736,
    'Arabidopsis' : 10.25131,
}

def main():
    ui.setup()
    maybe_draw_example()
    maybe_display_summary_statistics()

def get_selected_image():
    if Option_State['mode'] == 'View Example Images':
        image = download_example()
    if Option_State['mode'] == 'Upload An Image':
        image = get_uploaded_image()
    return image

def get_uploaded_image():
    return read_byte_stream(Option_State['uploaded_file'])

def download_example():
    urls = get_current_image_urls()
    return download_image(urls['image'] + '/download')

def get_current_image_urls():
    image_name = Option_State['image_name']
    return  Option_State['image_url_dicts'][image_name]

def download_annotations():
    urls = get_current_image_urls()
    return download_json(urls['annotations'] + '/download')

def download_predictions():
    urls = get_current_image_urls()
    return download_json(urls['predictions'] + '/download')

def download_ground_truth():
    urls = get_current_image_urls()
    return download_json(urls['ground_truth'] + '/download')

def draw_predictions(mpl_axis):
    predictions = get_and_filter_predictions()
    draw_labels_on_image(mpl_axis, predictions, False)

def get_and_filter_predictions():
    predictions = download_predictions()['detections']
    return filter_low_confidence_predictions(predictions)

def get_ground_truth():
    return download_ground_truth()['detections']

def filter_low_confidence_predictions(predictions):
    threshold = Option_State['confidence_threshold']
    predictions = [
        prediction for prediction in predictions 
        if prediction['confidence'] >= threshold
    ]
    return predictions

def draw_ground_truth(mpl_axis):
    ground_truth = get_ground_truth()
    draw_labels_on_image(mpl_axis, ground_truth, True)

def draw_labels_on_image(mpl_axis, annotations, gt):
    if Option_State['draw_bboxes']:
        draw.bboxes(mpl_axis, annotations, gt)
    if Option_State['draw_masks']:
        draw.masks(mpl_axis, annotations, gt)
    if Option_State['draw_keypoints']:
        draw.keypoints(mpl_axis, annotations, gt)

def maybe_draw_example():
    if do_draw():
        draw_example()

def do_draw():
    draw_check_1 = not is_mode_instructions() and is_file_uploaded()
    draw_check_2 = is_mode_view_examples()
    return  draw_check_1 or draw_check_2

def is_mode_instructions():
    return Option_State['mode'] == 'Instructions'

def is_file_uploaded():
    return Option_State['uploaded_file'] is not None

def is_mode_view_examples():
    return Option_State['mode'] == 'View Example Images'

def is_mode_upload_an_example():
    return Option_State['mode'] == 'Upload An Image'

def maybe_display_summary_statistics():
    if not is_mode_instructions():
        display_summary_statistics()

def display_summary_statistics():
    column_names, column_human, column_predicted = st.beta_columns(3)
    with column_names:
        display_summary_names()
    if not is_mode_upload_an_example():
        with column_human:
            display_ground_truth_summary_statistics()
        with column_predicted:
            display_prediction_summary_statistics()
    else:
        # Temporay while live inference is not added
        display_summary_names()

def display_summary_names():
    st.write("Properties")
    st.write("Stomata Count:")
    st.write("Stomatal Density:")
    st.write("Average Pore Length:")
    st.write("Average Pore Width:")
    st.write("Average Pore Area:")

def display_ground_truth_summary_statistics():
    ground_truth = get_ground_truth()
    st.write("Human Annotations")
    calculate_and_display_summary_statistics(ground_truth)

def display_prediction_summary_statistics():
    predictions = get_and_filter_predictions()
    st.write("Model Estimates")
    calculate_and_display_summary_statistics(predictions)

def calculate_and_display_summary_statistics(annotations):
    display_pore_count(annotations)
    display_pore_density(annotations)
    display_average_length(annotations)
    display_average_width(annotations)
    display_average_area(annotations)

def display_pore_count(annotations):
    st.write(f"{len(annotations)}")

def display_pore_density(annotations):
    area = IMAGE_AREA[Option_State['plant_type']]
    density = round(len(annotations) / area, 2)
    st.write(f"{density} stomata/mm\u00B2")

def display_average_length(annotations):
    average_length = average_key(annotations, 'length')
    st.write(f"{average_length} \u03BCm")

def display_average_width(annotations):
    average_width = average_key(annotations, 'width')
    st.write(f"{average_width} \u03BCm")

def display_average_area(annotations):
    average_area = average_key(annotations, 'area')
    st.write(f"{average_area} \u03BCm\u00B2")

def average_key(annotations, key):
    values = [ annotation[key] for annotation in annotations]
    average = sum(values) / len(values)
    average /= CAMERA_CALIBRATION[Option_State['plant_type']]
    return round(average, 2)

def draw_example():
    image = get_selected_image()
    image = preprocess_image(image)
    fig, ax = setup_plot(image)
    if Option_State['draw_predictions'] and is_mode_view_examples():
        draw_predictions(ax)
    if Option_State['draw_ground_truth'] and is_mode_view_examples():
        draw_ground_truth(ax)
    
    # Add Resize here
    st.write(fig)

def setup_plot(image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    return fig, ax

@st.cache(show_spinner=True)
def download_image(url):
    with urllib.request.urlopen(url) as response:
        image = read_byte_stream(response)
    return image
@st.cache(show_spinner=True)
def download_json(url):
    with urllib.request.urlopen(url) as response:
        downloaded_json = json.loads(response.read())
    return downloaded_json

def read_byte_stream(bytestream):
    return np.asarray(bytearray(bytestream.read()), dtype="uint8")

def preprocess_image(image):
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Make a resize function here size: (800,500)

if __name__ == '__main__':
    main()
