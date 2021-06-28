import streamlit as st

from interface import ui
from tools.load import maybe_create_folders

from app.example_images import maybe_draw_example
from app.example_output import maybe_show_slide_output_example
from app.inference import maybe_do_inference
from app.summary_statistics import maybe_display_summary_statistics

Is_Setup = False


def main():
    ui.setup()
    maybe_do_inference()
    maybe_draw_example()
    maybe_display_summary_statistics()
    maybe_show_slide_output_example()


if __name__ == "__main__":
    if not Is_Setup:
        # Current this runs each time a setting is changed
        maybe_create_folders()
        st.set_page_config(
            page_title="SAI - StomAI",
            page_icon=":purple_circle:",
            layout="centered",
            initial_sidebar_state="expanded",
        )
        from inference.modeling.stoma_head import KRCNNConvHead, KPROIHeads
        Is_Setup = True

    main()
