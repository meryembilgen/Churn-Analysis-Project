import streamlit as st
import pandas as pd

# import numpy as np
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go

# from sklearn.pipeline import Pipeline
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from xgboost import XGBClassifier
import catboost
from catboost import CatBoostClassifier

import pickle
from PIL import Image


img = Image.open("employee.jpg")

st.set_page_config(
    page_title='Predict',
    page_icon=img
)


st.markdown("""
			<div style="background-color:#23a5a0;\
						border-radius: 10px;\
						padding:15px">
			<h2 style="color:white;\
					   text-align:center;\
					   font-family:cursive;">Employee Churn Prediction\
			</h2>
			</div>
			""", unsafe_allow_html=True
			)

st.markdown("""
			<style>
    		[data-baseweb="select"] {
        							margin-top: 50px;
    								}
    		</style>
    		""", unsafe_allow_html=True,
			)


# Adding image
st.image("https://millidusunce.com/wp-content/uploads/2020/08/Bilgi-birikimi.jpg", width=700)


# Model Selection
st.markdown("""
			<center>
			<p style='font-size:150%;\
						font-family:cursive;\
						background-color:#23a5a0;\
						border-radius: 10px;\
						color:white;'>Select Your Model\
			</p>
			</center>
			""", unsafe_allow_html=True
			)

selection = st.selectbox("",["Logistic Regression", "SVC", 'KNN',"Random Forest", "XGBoost", 'CatBoost'])

if selection =="Logistic Regression":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'Logistic Regression'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('log_model', 'rb'))
elif selection =="SVC":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'SVC'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('svm_model', 'rb'))
elif selection =="KNN":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'KNN'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('knn_model', 'rb'))
elif selection =="Random Forest":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'Random Forest'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('rf_model', 'rb'))
elif selection =="XGBoost":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'XGBoost'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('xgb_model', 'rb'))
else:
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:#23a5a0;font-weight:bold'>\
				'CatBoost'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('catboost_model', 'rb'))
	

df = pd.read_csv("dataset.csv")
show_data = '<p style="font-family:cursive; color:#23a5a0; font-size: 20px;"><b>Show Data</b></p>'
st.markdown(show_data, unsafe_allow_html=True)
if st.checkbox(" ") :
    st.table(df.head())


st.sidebar.write('\n')

st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:#23a5a0; font-size:100%; font-family:cursive; border-radius: 10px;'>Please select features of employee\
			</p>
			""", unsafe_allow_html=True
			)


# To take feature inputs
satisfaction_level = st.sidebar.slider("Satisfation Level", 0.0, 1.0, 0.1)

last_evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.1)

number_project	= st.sidebar.slider("The Number of Projects", 2, 7, 4)

average_montly_hours = st.sidebar.slider('Average Monthly Hours', 96, 310, 200)

time_spend_company = st.sidebar.slider('Experience (Years)', 2, 10, 5)


# Create a dataframe using feature inputs
sample = {'satisfaction_level': satisfaction_level,
          'last_evaluation': last_evaluation,
          'number_project': number_project,
          'average_montly_hours': average_montly_hours,
          'time_spend_company': time_spend_company,
           }

df = pd.DataFrame.from_dict([sample])


employee_info = '<p style="font-family:cursive; color:#23a5a0; font-size: 20px;"><b>Employee Info</b></p>'
st.markdown(employee_info, unsafe_allow_html=True)

st.table(df)


predict_employee_churn = '<p style="font-family:cursive; color:#23a5a0; font-size: 20px;"><b>Predict Employee Churn</b></p>'
st.markdown(predict_employee_churn, unsafe_allow_html=True)


# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)

if predict:
    if result[0] == 1:
        st.error(f'Employee will LEAVE :(')
               
    else :
        st.success(f'Employee will STAY :)')
	
# fig = go.Figure(go.Indicator(  # probability gauge chart)
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         value = model.predict_proba(df)[0][1],
#         mode = "gauge+number+delta",
#         title = {'text': "Churn Probability"},
#         delta = {'reference': 0.5},
#         gauge = {'axis': {'range': [None, 1]},
#                  'steps' : [{'range': [0, 0.5], 'color': "red"},
#                             {'range': [0.5, 1], 'color': "green"}],
#                  'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}))

# st.plotly_chart(fig, use_container_width=True)  # to display chart on application page

# st.markdown("Thank you for visiting our **Employee Churn Prediction** page.")

import streamlit as st 
from PIL import Image

img = Image.open("employee.jpg")

st.set_page_config(
    page_title='Employee Churn Prediction',
    page_icon=img
)

st.text('This is a web app to predict employee churn.')

'''
## WELCOME!

Welcome to the "***Employee Churn Analysis Project***". This is the second project of the Capstone Project Series, where we created our classification models for various business environments. 

In this project we researched what is Employee Churn?, How it is different from customer churn, Exploratory data analysis and visualization of employee churn dataset using ***matplotlib*** and ***seaborn***, model building and evaluation using python ***scikit-learn*** and ***Tensorflow-Keras*** packages. 

We implement classification techniques in Python, Using Scikit-Learn, allowing you to successfully make predictions with Distance Based, Bagging, and Boosting algorithms for this project. On the other hand, for Deep Learning, we used Tensorflow-Keras. 

- NOTE: This project assumes that you already know the basics of coding in Python and are familiar with model deployement as well as the theory behind Distance Based, Bagging, Boosting algorithms, and Confusion Matrices.

---

'''                 

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