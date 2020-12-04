import json
import urllib

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import ui
import draw
from ui import OPTION_STATE

def main():
    ui.setup()
    maybe_draw_example()

def get_selected_image():
    if OPTION_STATE['mode'] == 'View Example Images':
        image = download_example()
    if OPTION_STATE['mode'] == 'Upload An Image':
        image = get_uploaded_image()
    return image

def get_uploaded_image():
    return read_byte_stream(OPTION_STATE['uploaded_file'])

def download_example():
    urls = get_current_image_urls()
    return download_image(urls['original'] + '/download')

def get_current_image_urls():
    image_name = OPTION_STATE['image_name']
    return  OPTION_STATE['image_url_dicts'][image_name]

def download_annotations():
    urls = get_current_image_urls()
    return download_json(urls['annotations'] + '/download')

def draw_predictions(mpl_axis):
    annotations = download_annotations()
    predictions = extract_predictions(annotations)
    mpl_axis = draw.bboxes(mpl_axis, predictions, False)
    return mpl_axis

def extract_predictions(annotations):
    detections = annotations['detections']
    return [ detection['pred'] for detection in detections ]

def extract_ground_truth(annotations):
    detections = annotations['detections']
    return [ detection['gt'] for detection in detections ]

def draw_ground_truth(mpl_axis):
    annotations = download_annotations()
    ground_truth = extract_ground_truth(annotations)
    mpl_axis = draw.bboxes(mpl_axis, ground_truth, True)
    return mpl_axis

def maybe_draw_example():
    if do_draw():
        draw_example()

def do_draw():
    draw_check_1 = not is_mode_instructions() and is_file_uploaded()
    draw_check_2 = is_mode_view_examples()
    return  draw_check_1 or draw_check_2

def is_mode_instructions():
    return OPTION_STATE['mode'] == 'Instructions'

def is_file_uploaded():
    return OPTION_STATE['uploaded_file'] is not None

def is_mode_view_examples():
    return OPTION_STATE['mode'] == 'View Example Images'

def draw_example():
    image = get_selected_image()
    image = preprocess_image(image)
    fig, ax = setup_plot(image)
    if OPTION_STATE['draw_predictions']:
        ax = draw_predictions(ax)
    if OPTION_STATE['draw_ground_truth']:
        ax = draw_ground_truth(ax)
    
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
