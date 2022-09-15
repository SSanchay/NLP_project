import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = '/home/a_ladin/ds-phase-0/learning/datasets/tips.csv'
tips = pd.read_csv(path, index_col=0)

st.sidebar.header('Параметры сортировки')
def choose_sex(df):
    sex = st.sidebar.selectbox('Сортировка по полу', ('Male and Female', 'Male', 'Female'))
    if sex == 'Male and Female':
        return df
    else:
        return df[df['sex'] == sex]


def choose_smoker(df):
    smoker = st.sidebar.selectbox('Сортировка по курению', ('Both', 'Yes', 'No'))
    if smoker == 'Both':
        return df
    else:
        return df[df['smoker'] == smoker]


def choose_day(df):
    day = st.sidebar.selectbox('Сортировка по дням недели', ('All days', 'Sun', 'Sat', 'Thur', 'Fri'))
    if day == 'All days':
        return df
    else:
        return df[df['day'] == day]


def choose_time(df):
    time = st.sidebar.selectbox('Сортировка по времени', (  'All times', 'Dinner', 'Lunch'))
    if time == 'All times':
        return df
    else:
        return df[df['time'] == time]


total_bill = st.sidebar.slider('Сортировка по размеру счета', 0, 50)
tip = st.sidebar.slider('Сортировка по размеру чаевых', 0, 9)

tips = choose_sex(tips)
tips = choose_smoker(tips)
tips = choose_time(tips)
tips = choose_day(tips)

st.write("""## Визуализация исследования чаевых!""")

tips = tips[tips['total_bill'] > total_bill]
tips = tips[tips['tip'] > tip]

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_title('Гистограмма, показывающая размер счета', size=35)
ax.set_xlabel('Заказы', size=35)
ax.set_ylabel('Размер счета', size=35)
ax.set_yticks(np.arange(0, 55, 5))
ax.set_xticks(np.arange(0, 245, 5))
ax.bar(x=tips.index, height=tips['total_bill'])
st.pyplot(fig)

st.write("""#### График, показывающий размер счета""")
st.line_chart(tips['total_bill'])
st.write("""#### График, показывающий размер чаевых!""")
st.line_chart(tips['tip'])
