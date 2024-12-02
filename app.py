import pandas as pd
import streamlit as st

# Default page configuration
st.set_page_config(
    page_title='Funds Receivable Summary',
    page_icon=':bar_chart:',
    layout="wide")

# Hide Streamlit menus
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Page Title
st.title('Funds Receivable Summary')

# Adding Sidebar for Input files upload
with st.sidebar:

    # Upload Transaction file
    st.header('Upload the Transaction file')
    transaction = st.file_uploader('Upload the Transaction file from ERP', type='csv', label_visibility='collapsed')

    # Upload Transaction file
    st.header('Upload the WBS file')
    wbs = st.file_uploader('Upload the WBS file from ERP', type='csv', label_visibility='collapsed')