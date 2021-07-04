from interface.upload_folder import image_folder_text_box
import json
import os

from tools.state import Option_State


def get_current_image_urls():
    image_name = Option_State["image_name"]
    return Option_State["image_url_dicts"][image_name]


def load_json(filepath):
    with open(filepath, "r") as file:
        json_dict = json.load(file)
    return json_dict


# Help functions to check which mode the application is in
def is_drawing_mode():
    return not (
        is_mode_instructions() or
        is_mode_slide_output_example() or
        is_mode_upload_multiple_images()
    )


def is_file_uploaded():
    return Option_State["uploaded_file"] is not None


def is_image_folder_avaiable():
    return not Option_State["folder_path"] == ""


def is_mode_view_examples():
    return Option_State["mode"] == "View Example Images"


def is_mode_instructions():
    return Option_State["mode"] == "Instructions"


def is_mode_upload_an_example():
    return Option_State["mode"] == "Upload An Image"


def is_mode_slide_output_example():
    return Option_State["mode"] == "View Example Slide Output"


def is_mode_upload_multiple_images():
    return Option_State["mode"] == "Upload Multiple Images"
