import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("employee.jpg")

st.set_page_config(
    page_title='HR Dataset',
    page_icon=img
)


hr_dataset = '<center> <p style="font-family:cursive; border-radius: 10px; background-color:#23a5a0; color:white; font-size: 40px;"><b>HR Dataset</b></p> </center>'
st.markdown(hr_dataset, unsafe_allow_html=True)

'''
The HR dataset has 14,999 samples. In the given dataset, you have two types of employee one who stayed and another who left the company.

You can describe 10 attributes in detail as:
- ***satisfaction_level:*** It is employee satisfaction point, which ranges from 0-1.
- ***last_evaluation:*** It is evaluated performance by the employer, which also ranges from 0-1.
- ***number_projects:*** How many of projects assigned to an employee?
- ***average_monthly_hours:*** How many hours in averega an employee worked in a month?
- ***time_spent_company:*** time_spent_company means employee experience. The number of years spent by an employee in the company.
- ***work_accident:*** Whether an employee has had a work accident or not.
- ***promotion_last_5years:*** Whether an employee has had a promotion in the last 5 years or not.
- ***Departments:*** Employee's working department/division.
- ***Salary:*** Salary level of the employee such as low, medium and high.
- ***left:*** Whether the employee has left the company or not.

---

'''

df = pd.read_csv('dataset.csv')

data_header = '<p style="font-family:cursive; color:#23a5a0; font-size: 25px;"><b>Data Header</b></p>'
st.markdown(data_header, unsafe_allow_html=True)
st.write(df.head())

statistics = '<p style="font-family:cursive; color:#23a5a0; font-size: 25px;"><b>Data Statistics</b></p>'
st.markdown(statistics, unsafe_allow_html=True)
st.write(df.describe())

feature = df.columns
features = st.selectbox('Select Feature', feature)

css = '''
<style>
    .stSelectbox [data-testid='stMarkdownContainer'] {
        color: #23a5a0;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

if features:
    st.text(f'{features} distribution on the HR dataset')
    model_dist = pd.DataFrame(df[features].value_counts())
    st.bar_chart(model_dist, height=400, width=600 , use_container_width=False)