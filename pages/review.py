import streamlit as st
import pandas as pd


user_input = st.text_input("Напиши начало строки, все остальное я сделаю сама")
st.write(user_input)

