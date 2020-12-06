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
    'draw_predictions' : False,
    'draw_ground_truth' : False,
    'draw_bboxes' : True,
    'draw_masks' : True,
    'draw_keypoints' : True,
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
    drawing_options()

def drawing_options():
    draw_ground_truth_checkbox()
    draw_predictions_checkbox()
    if is_drawing_enabled():
        st.sidebar.write('Types of Measurments:')
        draw_bboxes_checkbox()
        draw_masks_checkbox()
        draw_keypoints_checkbox()

def draw_bboxes_checkbox():
    OPTION_STATE['draw_bboxes'] = st.sidebar.checkbox(
        'Show Bounding Boxes',
        value=True,
    )
def draw_masks_checkbox():
    OPTION_STATE['draw_masks'] = st.sidebar.checkbox(
        'Show Pore Segmentations',
        value=True,
    )
def draw_keypoints_checkbox():
    OPTION_STATE['draw_keypoints'] = st.sidebar.checkbox(
        'Show Lengths and Widths',
        value=True,
    )

def is_drawing_enabled():
    drawing_predictions = OPTION_STATE['draw_predictions']
    drawing_ground_truth = OPTION_STATE['draw_ground_truth']
    return drawing_predictions or drawing_ground_truth 

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