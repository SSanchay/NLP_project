import streamlit as st

import base64
def add_bg_from_local(image_file):
    with open(image_file, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f'''
    <style>
    .stApp {{
        background-image: url(data:image/{'png'};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    ''',
    unsafe_allow_html=True
    )
add_bg_from_local('generator.jpeg')

def main_page():
    st.markdown("# Main page")
    st.sidebar.markdown("# Main page")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3")
    st.sidebar.markdown("# Page 3")


