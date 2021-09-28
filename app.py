import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# for manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
plt.style.use("dark_background")
#sns.set_style('whitegrid')

# to filter warnings
import warnings
warnings.filterwarnings('ignore')

# for interactivity
from ipywidgets import interact

st.title("Agricultural Production Optimization Engine")

# Reading the dataset
data= pd.read_csv('data.csv')

x= data.drop(['label'], axis=1)
y= data['label']

# let's create training and testing sets for validation of results
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

# let's create predictive model
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(x_train,y_train)

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
random_forest= RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

from sklearn.tree import DecisionTreeClassifier
DecTree= DecisionTreeClassifier()
DecTree.fit(x_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier()
KNN.fit(x_train, y_train)

from sklearn.naive_bayes import GaussianNB
NB= GaussianNB()
NB.fit(x_train, y_train)

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)

Nv = st.sidebar.radio("Navigator", ["Home","Prediction","Contribute"])
if Nv== "Home":
    #st.write("### Home")
    st.image("app.png", width= 700)
    if st.checkbox("Show Dataset"):
        st.table(data)

    st.subheader("\nSoil Requirement of Each Crop")
    if st.checkbox("Show Soil Requirement Graphs"):
        condition = st.selectbox("Conditions",['Nitrogen Requirement','Phosphorous Requirement','Potassium Requirement','Temperature Requirement',
                                               'PH Requirement','Humidity Requirement','Rainfall Requirement'])
        if condition == "Nitrogen Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["N"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Nitrogen Requirement", fontsize=12)
            st.pyplot()
        if condition == "Phosphorous Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["P"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Phosphorous Requirement", fontsize=12)
            st.pyplot()
        if condition == "Potassium Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["K"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Potassium Requirement", fontsize=12)
            st.pyplot()
        if condition == "Temperature Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["temperature"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Temperature Requirement", fontsize=12)
            st.pyplot()
        if condition == "Humidity Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["humidity"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Humidity Requirement", fontsize=12)
            st.pyplot()
        if condition == "PH Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["ph"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("PH Requirement", fontsize=12)
            st.pyplot()
        if condition == "Rainfall Requirement":
            plt.figure(figsize=(5, 3))
            sns.barplot(data['label'], data["rainfall"])
            plt.xlabel('\nCrops', fontsize=14)
            plt.xticks(rotation=90)
            plt.ylabel("Rainfall Requirement", fontsize=12)
            st.pyplot()

    st.subheader("\nDistribution of Agricultural Conditions")
    if st.checkbox("Show Distribution Graphs"):
        con = st.selectbox("Conditions",['N','P','K','Temperature','PH','Humidity','Rainfall'])
        if con == "N":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["N"])
            plt.xlabel("\nNitrogen", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["N"].min(), color='y', label='Minimum')
            plt.axvline(data["N"].mean(), color='orange', label='Mean')
            plt.axvline(data["N"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "P":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["P"])
            plt.xlabel("\nPhosphourous", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["P"].min(), color='y', label='Minimum')
            plt.axvline(data["P"].mean(), color='orange', label='Mean')
            plt.axvline(data["P"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "K":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["K"])
            plt.xlabel("\nPotassium", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["K"].min(), color='y', label='Minimum')
            plt.axvline(data["K"].mean(), color='orange', label='Mean')
            plt.axvline(data["K"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "Temperature":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["temperature"])
            plt.xlabel("\nTemperature", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["temperature"].min(), color='y', label='Minimum')
            plt.axvline(data["temperature"].mean(), color='orange', label='Mean')
            plt.axvline(data["temperature"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "PH":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["ph"])
            plt.xlabel("\nPH", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["ph"].min(), color='y', label='Minimum')
            plt.axvline(data["ph"].mean(), color='orange', label='Mean')
            plt.axvline(data["ph"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "Humidity":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["humidity"])
            plt.xlabel("\nHumidity", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["humidity"].min(), color='y', label='Minimum')
            plt.axvline(data["humidity"].mean(), color='orange', label='Mean')
            plt.axvline(data["humidity"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()
        if con == "Rainfall":
            plt.figure(figsize=(5, 3))
            sns.distplot(data["rainfall"])
            plt.xlabel("\nRainfall", fontsize=14)
            plt.ylabel('Density',fontsize=14)
            plt.axvline(data["rainfall"].min(), color='y', label='Minimum')
            plt.axvline(data["rainfall"].mean(), color='orange', label='Mean')
            plt.axvline(data["rainfall"].max(), color='grey', label='Maximum')
            plt.legend()
            st.pyplot()

if Nv == "Prediction":

    st.subheader("\nCrop Predictor\n")
    N = st.number_input("\nNitrogen Value: ",50.00, step=0.10)
    P = st.number_input("Phosphorous Value: ", 50.00 ,step=0.10)
    K = st.number_input("Potassium Value: ", 50.00 ,step=0.10)
    T = st.number_input("Tempreture: ", 25.00 ,step=0.10)
    H = st.number_input("Humidity: ", 50.00 ,step=0.10)
    PH = st.number_input("PH Value: ", 7.00 ,step=0.10)
    R = st.number_input("Rainfall: ", 200.00 ,step=0.10)

    st.write("\n\n\n")
    op=st.selectbox("Choose ML Algorithm",['Random Forest','Logistic Regression', 'Decision Tree','KNN', 'Naive Bayes', 'SVM'])
    

    st.write("\n\n\n")
    if st.button("Predict"):

        if op=="Logistic Regression":
            y_pred_LR= LogReg.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using Logistic Regression is:")
            st.success(y_pred_LR)
        
        if op=="Random Forest":
            y_pred_RF= random_forest.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using Random Forest is:")
            st.success(y_pred_RF)

            
        if op=="Decision Tree":

            y_pred_DT= DecTree.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using Decision Tree is:")
            st.success(y_pred_DT)

        if op=="KNN":
            y_pred_KNN= DecTree.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using KNN is:")
            st.success(y_pred_KNN)
            
        if op=="Naive Bayes":
            y_pred_NB= NB.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using Naive Bayes is:")
            st.success(y_pred_NB)
            
        if op=="SVM":
            y_pred_SVM= svm.predict([[N, P, K, T, H, PH, R]])    
            st.subheader(f"\nPredicted Crop by using SVM is:")
            st.success(y_pred_SVM)

if Nv == "Contribute":
    st.subheader("Contribute to our Dataset")
    N = st.number_input("Nitrogen Value: ", 0.00, 150.00, 50.00, step=0.5)
    P = st.number_input("Phosphorous Value: ", 0.00, 150.00, 50.00, step=0.5)
    K = st.number_input("Potassium Value: ", 0.00, 120.00, 50.00, step=0.5)
    T = st.number_input("Tempreture: ", 0.00, 60.00, 25.00, step=0.5)
    H = st.number_input("Humidity: ", 10.00, 100.00, 50.00, step=0.5)
    PH = st.number_input("PH Value: ", 0.00, 10.00, 7.00, step=0.5)
    R = st.number_input("Rainfall: ", 20.00, 300.00, 200.00, step=0.5)
    crop = st.text_input("Crop: ")

    if st.button("Contribute"):
        to_add= {"N":[N], "P":[P], "K":[K], "temperature":[T], "humidity":[H], "ph":[PH], "rainfall":[R], "label":[crop]}
        to_add= pd.DataFrame(to_add)
        to_add.to_csv("app.csv", mode='a', header=False, index=False)
        st.success("Thanks for Your Contribution")
