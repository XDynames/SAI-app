from PIL import Image

import streamlit as st


def display_instructions():
    with open("instructions.md") as file:
        markdown_string = file.read()
    st.markdown(markdown_string, unsafe_allow_html=True)
    setup_footer()


def setup_footer():
    display_logos()


def dispaly_links():
    columns = st.columns(3)
    with columns[0]:
        st.markdown("[Publication]()")
    with columns[1]:
        st.markdown("[Application Code](https://github.com/XDynames/SAI-app)")
    with columns[2]:
        st.markdown(
            "[Deep Learning Code](https://github.com/XDynames/SAI-training)"
        )


def display_logos():
    columns = st.columns(3)
    with columns[0]:
        st.image(Image.open("logos/peb.jpeg"), width=145)
    with columns[1]:
        st.image(Image.open("logos/uoa.jpeg"), width=140)
    with columns[2]:
        st.image(Image.open("logos/aiml.png"), width=275)
