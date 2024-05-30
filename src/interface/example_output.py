import streamlit as st


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
    st.write("### Pore Measurement Output")
    message = (
        "Below is an Example of .csv file output for model"
        " predictions on images. In this example, each row"
        " contains measurements for a different pore. "
        "Class is the predicted state of the pore."
    )
    st.write(message)
