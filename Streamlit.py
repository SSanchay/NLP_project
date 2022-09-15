import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = '/home/a_ladin/ds-phase-0/learning/datasets/tips.csv'
tips = pd.read_csv(path, index_col=0)


def main_page():
    st.markdown("# Main page")
    st.sidebar.markdown("# Main page")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3")
    st.sidebar.markdown("# Page 3")


