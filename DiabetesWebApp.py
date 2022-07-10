# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:26:50 2022

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('AnshulX/Diabetes-Prediction-Model/trained_model.sav','rb')) 

def diabeticPrediction(input_data):
    input_data=(1,89,66,23,94,28.1,0.167,21)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    standardized_input_data=input_data_reshaped
    print(standardized_input_data)
    prediction = loaded_model.predict(standardized_input_data)
    print(prediction) 
    if (prediction[0]==0):
        print('The person is not Diabetic')
    else:
        print('The person is Diabetic')
        
def main():
    #giving the title
    st.title('Diabetes Prediction Web App')
    
    Pregnancies =st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabeticPrediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
