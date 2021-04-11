import streamlit as st

from interface import ui

from app.example_images import maybe_draw_example
from app.example_output import maybe_show_slide_output_example
from app.summary_statistics import maybe_display_summary_statistics

is_setup = False


def main():
    maybe_setup()
    maybe_draw_example()
    maybe_display_summary_statistics()
    maybe_show_slide_output_example()


def maybe_setup():
    global is_setup
    if not is_setup:
        ui.setup()
        is_setup = True


if __name__ == "__main__":
    st.set_page_config(
        page_title="SAI - StomAI",
        page_icon=":purple_circle:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    main()
