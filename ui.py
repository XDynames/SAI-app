import streamlit as st

from cloud_files import IMAGE_DICTS, EXTERNAL_DEPENDANCIES


PLANT_OPTIONS = [
    'Barley',
    'Arabidopsis',
]

OPTION_STATE = {
    'mode' : '',
    'plant_type' : '',
    'image_url_dicts' : {},
    'image_name' : '',
    'draw_predictions' : '',
    'draw_ground_truth' : '',
    'uploaded_file' : None,
}

def setup():
    setup_heading()
    setup_sidebar()

def setup_heading():
    st.title('Automatic Stoma Measurement')
    st.subheader('Accelerating plant physiology research')

def setup_sidebar():
    mode_selection()
    if OPTION_STATE['mode'] == 'Instructions':
        display_instructions()
    if OPTION_STATE['mode'] == 'Upload An Image':
        file_upload()
    if OPTION_STATE['mode'] == 'View Example Images':
        predefined_example_selections()

def predefined_example_selections():
    plant_type_selection()
    image_selection()
    draw_ground_truth_checkbox()
    draw_predictions_checkbox()

def file_upload():
    OPTION_STATE['uploaded_file'] = st.file_uploader(
        "Upload Files",
        type=['png']
    )

def mode_selection():
    OPTION_STATE['mode'] = st.sidebar.selectbox(
        'Select Application Mode:',
        [ 'Instructions', 'View Example Images', 'Upload An Image' ]
    )

def display_instructions():
    st.write('Put application instructions here')

def plant_type_selection():
    OPTION_STATE['plant_type'] = st.sidebar.selectbox(
        'Select a plant type:',
        PLANT_OPTIONS
    )

def image_selection():
    image_dict = IMAGE_DICTS[OPTION_STATE['plant_type']]
    OPTION_STATE['image_url_dicts'] = image_dict
    OPTION_STATE['image_name'] = st.sidebar.selectbox(
        'Select image:',
        [ image_name for image_name in image_dict.keys() ]
    )

def draw_ground_truth_checkbox():
    OPTION_STATE['draw_ground_truth'] = st.sidebar.checkbox(
        'Show Human Measurements'
    )

def draw_predictions_checkbox():
    OPTION_STATE['draw_predictions'] = st.sidebar.checkbox(
        'Show Model Predictions'
    )