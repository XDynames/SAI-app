from PIL import Image

import streamlit as st

from cloud_files import IMAGE_DICTS

IS_ONLINE = True

PLANT_OPTIONS = [
    "Barley",
    "Arabidopsis",
]

IMAGE_AREA = {
    "Barley": 0.3229496,
    "Arabidopsis": 0.04794822,
}
# pixels / micron
CAMERA_CALIBRATION = {
    "Barley": 4.2736,
    "Arabidopsis": 10.25131,
}

OPENCV_FILE_SUPPORT = ["png", "bmp", "jpeg", "jpg", "jpe", "jp2", "tiff", "tif"]

ENABLED_MODES = [
    "Instructions",
    "View Example Images",
    "View Example Slide Output",
    "View Example Group Output",
    "Upload An Image",
]

Option_State = {
    "mode": "",
    "plant_type": "",
    "image_url_dicts": {},
    "image_name": "",
    "draw_predictions": False,
    "draw_ground_truth": False,
    "draw_bboxes": True,
    "draw_masks": True,
    "draw_keypoints": True,
    "uploaded_file": None,
    "confidence_threshold": 0.5,
    "image_size": None,
    "image_area": 0.0,
    "camera_calibration": None,
}


def setup():
    draw_title()
    setup_sidebar()


def setup_heading():
    columns = st.beta_columns(3)
    with columns[0]:
        st.image(Image.open("logos/peb.jpeg"), width=145)
    with columns[1]:
        st.image(Image.open("logos/uoa.jpeg"), width=140)
    with columns[2]:
        st.image(Image.open("logos/aiml.png"), width=275)
    subheading = (
        "<h3 style='text-align: center'>Accelerating plant" " physiology research</h3>"
    )
    st.markdown(subheading, unsafe_allow_html=True)


def draw_title():
    heading = "<h1 style='text-align: center'>SAI - StomaAI</h1>"
    st.markdown(heading, unsafe_allow_html=True)


def setup_sidebar():
    mode_selection()
    MODE_METHODS[Option_State["mode"]]()


def display_example_selection():
    plant_type_selection()
    image_selection()
    show_calibration_information()
    drawing_options()


def display_upload_image():
    if IS_ONLINE:
        message = (
            "This feature is disabled in the online version of"
            " the application. To use this functionality please"
            " go to <INSERT LINK> to install and run the application"
            " locally."
        )
        st.write(message)
    else:
        file_upload()
        draw_calibration_textboxes()


def display_group_output_example():
    message = (
        "Below is an Example of .csv file output for model"
        " predictions on a treatment group. In this case "
        "the arabidopsis and barley samples are used as stand"
        " ins for two different treatment groups. The rows will"
        " indicate which group the summary statistics belong to"
        " and columns will contain which measurements a summary"
        " relates to. "
    )
    st.write(message)


def display_slide_output_example():
    message = (
        "Below is an Example of .csv file output for model"
        " predictions on images. In this example, each row"
        " contains measurements for a different pore. "
        "Category id is the predicted state of the pore;"
        " 0 for closed, 1 for open. "
    )
    st.write(message)


def display_instructions():
    setup_heading()
    with open("instructions.md") as file:
        markdown_string = file.read()
    st.markdown(markdown_string)


def drawing_options():
    draw_ground_truth_checkbox()
    draw_predictions_checkbox()
    if draw_predictions_enabled():
        confience_sliderbar()
    if drawing_enabled():
        st.sidebar.write("Types of Measurments:")
        draw_bboxes_checkbox()
        draw_masks_checkbox()
        draw_keypoints_checkbox()


def draw_bboxes_checkbox():
    Option_State["draw_bboxes"] = st.sidebar.checkbox(
        "Show Bounding Boxes",
        value=True,
    )


def draw_masks_checkbox():
    Option_State["draw_masks"] = st.sidebar.checkbox(
        "Show Pore Segmentations",
        value=True,
    )


def draw_keypoints_checkbox():
    Option_State["draw_keypoints"] = st.sidebar.checkbox(
        "Show Lengths and Widths",
        value=True,
    )


def show_calibration_information():
    print_camera_calibration()


def draw_calibration_textboxes():
    draw_camera_calibration_textbox()


def draw_camera_calibration_textbox():
    Option_State["camera_calibration"] = st.sidebar.number_input(
        "Camera Calibration (px/\u03BCm):",
        min_value=0.0,
        value=0.0,
        step=0.5,
        format="%.3f",
    )
    if Option_State["image_size"] is not None:
        image_size = Option_State["image_size"]
        area = convert_to_SIU_length(image_size[0]) * convert_to_SIU_length(
            image_size[1]
        )
        Option_State["image_area"] = area


def convert_to_SIU_length(pixel_length):
    return pixel_length * Option_State["camera_calibration"]


def print_camera_calibration():
    camera_calibration = CAMERA_CALIBRATION[Option_State["plant_type"]]
    Option_State["camera_calibration"] = camera_calibration
    message = f"Camera Calibration: {camera_calibration:.4} px/\u03BCm"
    st.sidebar.write(message)


def drawing_enabled():
    drawing_ground_truth = Option_State["draw_ground_truth"]
    return draw_predictions_enabled() or drawing_ground_truth


def draw_predictions_enabled():
    return Option_State["draw_predictions"]


def file_upload():
    Option_State["uploaded_file"] = st.file_uploader(
        "Upload Files", type=OPENCV_FILE_SUPPORT
    )


def mode_selection():
    Option_State["mode"] = st.sidebar.selectbox(
        "Select Application Mode:", ENABLED_MODES
    )


def confience_sliderbar():
    Option_State["confidence_threshold"] = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


def plant_type_selection():
    Option_State["plant_type"] = st.sidebar.selectbox(
        "Select a plant type:", PLANT_OPTIONS
    )


def image_selection():
    image_dict = IMAGE_DICTS[Option_State["plant_type"]]
    Option_State["image_url_dicts"] = image_dict
    Option_State["image_name"] = st.sidebar.selectbox(
        "Select image:", [image_name for image_name in image_dict.keys()]
    )


def draw_ground_truth_checkbox():
    Option_State["draw_ground_truth"] = st.sidebar.checkbox("Show Human Measurements")


def draw_predictions_checkbox():
    Option_State["draw_predictions"] = st.sidebar.checkbox("Show Model Predictions")


MODE_METHODS = {
    "Instructions": display_instructions,
    "Upload An Image": display_upload_image,
    "View Example Images": display_example_selection,
    "View Example Slide Output": display_slide_output_example,
    "View Example Group Output": display_group_output_example,
}
