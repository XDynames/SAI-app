from PIL import Image

import streamlit as st


def display_instructions():
    setup_heading()
    with open("instructions.md") as file:
        markdown_string = file.read()
    st.markdown(markdown_string)


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
