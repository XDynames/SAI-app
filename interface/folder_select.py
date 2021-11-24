import os

import streamlit as st

from tools.state import Option_State


def display_folder_selection():
    display_path_and_options()
    display_folder_buttons()


def display_path_and_options():
    columns = st.columns((6, 1, 1))
    with columns[0]:
        st.text_input(
                "Currently selected folder to measure images in:",
                value=Option_State['current_path']
            )
    with columns[1]:
        st.button('\u2191Folder', on_click=move_up_folder_callback)
        st.button('Submit', on_click=submit_button_callback)
    with columns[2]:
        st.button('Undo', on_click=undo_button_callback)
        st.button('Reset', on_click=reset_path_button_callback)
    st.write('Select a folder below to move to it:')


def display_folder_buttons():
    current_path = Option_State['current_path']
    columns = st.columns((1, 1))
    contents = os.listdir(current_path)
    i = 0
    for content in contents:
        if is_a_folder(current_path, content) and not is_hidden_file(content):
            with columns[i % 2]:
                st.button(
                    content,
                    on_click=folder_button_callback,
                    args=(content,)
                )
            i += 1


def is_a_folder(current_path: str, content: str) -> bool:
    return os.path.isdir(os.path.join(current_path, content))


def is_hidden_file(content: str) -> bool:
    return '.' == content[0]


def folder_button_callback(button_name: str):
    Option_State['current_path'] = os.path.join(
        Option_State['current_path'],
        button_name
    )


def undo_button_callback():
    tmp_path = Option_State['current_path']
    Option_State['current_path'] = Option_State['last_path']
    Option_State['last_path'] = tmp_path


def reset_path_button_callback():
    tmp_path = Option_State['current_path']
    Option_State['current_path'] = os.path.dirname(os.getcwd())
    Option_State['last_path'] = tmp_path


def submit_button_callback():
    Option_State['select_folder'] = False


def move_up_folder_callback():
    if not is_root():
        Option_State['last_path'] = Option_State['current_path']
        Option_State['current_path'] = os.path.dirname(
            Option_State['current_path']
        )


def is_root():
    return Option_State['current_path'] == ''
