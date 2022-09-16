import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colour import Color
import matplotlib.colors as mcolors

from sentiment_models import sentiment_RNN, sentiment_ML, sentiment_BERT_ML

st.set_page_config(
    page_title="Film review predictions",
    page_icon="👋",
)

st.title("Film review predictions")
st.sidebar.success("Select NLP project.")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

my_input = st.text_input("", st.session_state["my_input"])
submit = st.button("Submit review")
if submit:
    st.session_state["my_input"] = my_input
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # red = Color("red")
    # colors = list(red.range_to(Color("green"),5))
    # colors = [color.rgb for color in colors]

    clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
         (0.7, "green"), (0.75, "blue"), (1, "blue")]
    rvb = mcolors.LinearSegmentedColormap.from_list("", clist)

    model = ('Classic ML', 'LSTM', 'BERT')
    y_pos = np.arange(len(model))
    prob = np.array([sentiment_ML(my_input),sentiment_RNN(my_input),sentiment_BERT_ML(my_input)])

    fig.set_figwidth(4)
    fig.set_figheight(1)
    
    ax.barh(y_pos, prob, align='center', color=rvb(prob))
    ax.set_yticks(y_pos, labels=model)
    ax.set_xlim(left=0, right=1)
    ax.invert_yaxis()
    ax.set_xlabel('Positivity')
    ax.set_title('Model predictions')

    st.pyplot(fig)

    # st.write("Classic ML prediction: ", my_input)
