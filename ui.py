import streamlit as st

from cloud_files import IMAGE_DICTS, EXTERNAL_DEPENDANCIES


PLANT_OPTIONS = [
    'Barley',
    'Arabidopsis',
]

Option_State = {
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
    if Option_State['mode'] == 'Instructions':
        display_instructions()
    if Option_State['mode'] == 'Upload An Image':
        file_upload()
    if Option_State['mode'] == 'View Example Images':
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
    Option_State['draw_bboxes'] = st.sidebar.checkbox(
        'Show Bounding Boxes',
        value=True,
    )
def draw_masks_checkbox():
    Option_State['draw_masks'] = st.sidebar.checkbox(
        'Show Pore Segmentations',
        value=True,
    )
def draw_keypoints_checkbox():
    Option_State['draw_keypoints'] = st.sidebar.checkbox(
        'Show Lengths and Widths',
        value=True,
    )

def is_drawing_enabled():
    drawing_predictions = Option_State['draw_predictions']
    drawing_ground_truth = Option_State['draw_ground_truth']
    return drawing_predictions or drawing_ground_truth 

def file_upload():
    Option_State['uploaded_file'] = st.file_uploader(
        "Upload Files",
        type=['png']
    )

def mode_selection():
    Option_State['mode'] = st.sidebar.selectbox(
        'Select Application Mode:',
        [ 'Instructions', 'View Example Images', 'Upload An Image' ]
    )

def display_instructions():
    st.write('Put application instructions here')

def plant_type_selection():
    Option_State['plant_type'] = st.sidebar.selectbox(
        'Select a plant type:',
        PLANT_OPTIONS
    )

def image_selection():
    image_dict = IMAGE_DICTS[Option_State['plant_type']]
    Option_State['image_url_dicts'] = image_dict
    Option_State['image_name'] = st.sidebar.selectbox(
        'Select image:',
        [ image_name for image_name in image_dict.keys() ]
    )

def draw_ground_truth_checkbox():
    Option_State['draw_ground_truth'] = st.sidebar.checkbox(
        'Show Human Measurements'
    )

def draw_predictions_checkbox():
    Option_State['draw_predictions'] = st.sidebar.checkbox(
        'Show Model Predictions'
    )