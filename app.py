import urllib

import cv2
import numpy as np
import streamlit as st

import ui
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
    image_name = OPTION_STATE['image_name']
    urls = OPTION_STATE['image_url_dicts'][image_name]
    return download_image(urls['original'] + '/download')

def draw_predictions(image):
    image_name = OPTION_STATE['image_name']
    urls = OPTION_STATE['image_url_dicts'][image_name]
    url = urls['visualised'] + '/download'
    return download_image(url)

def draw_ground_truth(image):
    return image

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
    if OPTION_STATE['draw_predictions']:
        image = draw_predictions(image)
    if OPTION_STATE['draw_ground_truth']:
        image = draw_ground_truth(image)
    image = preprocess_image(image)
    st.image(image)

@st.cache(show_spinner=True)
def download_image(url):
    with urllib.request.urlopen(url) as response:
        image = read_byte_stream(response)
    return image

def read_byte_stream(bytestream):
    return np.asarray(bytearray(bytestream.read()), dtype="uint8")

def preprocess_image(image):
    image = cv2.resize(cv2.imdecode(image, cv2.IMREAD_COLOR), (800,500))
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

if __name__ == '__main__':
    main()
