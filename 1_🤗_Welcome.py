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

