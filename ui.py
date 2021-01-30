from PIL import Image
import streamlit as st

from cloud_files import IMAGE_DICTS, EXTERNAL_DEPENDANCIES

PLANT_OPTIONS = [
    'Barley',
    'Arabidopsis',
]

IMAGE_AREA = {
    'Barley' : 0.3229496,
    'Arabidopsis' : 0.04794822,
}
# pixels / micron
CAMERA_CALIBRATION = {
    'Barley' : 4.2736,
    'Arabidopsis' : 10.25131,
}

OPENCV_FILE_SUPPORT = [
    'png', 'bmp', 'jpeg', 'jpg',
    'jpe', 'jp2', 'tiff', 'tif'
]

ENABLED_MODES = [
    'Instructions',
    'View Example Images',
    'View Example Output',
    'Upload An Image'
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
    'confidence_threshold' : 0.5,
    'image_area' : None,
    'camera_calibration' : None,
}

def setup():
    setup_heading()
    setup_sidebar()

def setup_heading():
    columns = st.beta_columns(4)
    with columns[1]:
        st.image(Image.open("logos/peb.jpeg"), width=145)
    with columns[2]:
        st.image(Image.open("logos/uoa.jpeg"), width=140)
    with columns[3]:
        st.image(Image.open("logos/aiml.jpeg"), width=100)
    with columns[0]:
        heading = "<h1 style='text-align: center'>SAI <br> Stoma AI</h1>"
        st.markdown(heading, unsafe_allow_html=True)
    subheading = "<h3 style='text-align: center'>Accelerating plant physiology research</h3>"
    st.markdown(subheading, unsafe_allow_html=True)

def setup_sidebar():
    mode_selection()
    MODE_METHODS[Option_State['mode']]()

def display_example_selection():
    plant_type_selection()
    image_selection()
    show_calibration_information()
    drawing_options()

def display_upload_image():
    file_upload()
    draw_calibration_textboxes()

def display_example_output():
    st.write("Example of .csv file output for model predictions on images:")

def display_instructions():
    with open('instructions.md') as file:
        markdown_string = file.read()
    st.markdown(markdown_string)

def drawing_options():
    draw_ground_truth_checkbox()
    draw_predictions_checkbox()
    if draw_predictions_enabled():
        confience_sliderbar()
    if drawing_enabled():
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

def show_calibration_information():
    print_image_area_textbox()
    print_camera_calibration()
    
def draw_calibration_textboxes():
    draw_image_area_textbox()
    draw_camera_calibration_textbox()

def draw_camera_calibration_textbox():
    Option_State['camera_calibration'] = st.sidebar.number_input(
        "Camera Calibration (px/\u03BCm):",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format='%.3f'
    )

def draw_image_area_textbox():
    Option_State['image_area'] = st.sidebar.number_input(
        "Image Area (mm\u00B2):",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format='%.3f'
    )

def print_camera_calibration():
    camera_calibration = CAMERA_CALIBRATION[Option_State['plant_type']]
    Option_State['camera_calibration'] = camera_calibration
    message = f"Camera Calibration: {camera_calibration:.4} px/\u03BCm"
    st.sidebar.write(message)

def print_image_area_textbox():
    image_area = IMAGE_AREA[Option_State['plant_type']]
    Option_State['image_area'] = image_area
    st.sidebar.write(f"Image Area: {image_area:.3} mm\u00B2")

def drawing_enabled():
    drawing_ground_truth = Option_State['draw_ground_truth']
    return draw_predictions_enabled() or drawing_ground_truth 

def draw_predictions_enabled():
    return Option_State['draw_predictions']

def file_upload():
    Option_State['uploaded_file'] = st.file_uploader(
        "Upload Files",
        type=OPENCV_FILE_SUPPORT
    )

def mode_selection():
    Option_State['mode'] = st.sidebar.selectbox(
        'Select Application Mode:',
        ENABLED_MODES
    )

def confience_sliderbar():
    Option_State['confidence_threshold'] = st.sidebar.slider(
        'Confidence Threshold',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

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

MODE_METHODS = {
    'Instructions' : display_instructions,
    'Upload An Image' : display_upload_image,
    'View Example Images' : display_example_selection,
    'View Example Output' : display_example_output,
}