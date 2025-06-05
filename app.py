
# Your streamlit code goes here
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

# Load trained models and other data
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
symptom_index = pickle.load(open('symptom_index.pkl', 'rb'))

# UI
st.title("Disease Prediction System")
st.write("Enter your symptoms (separated by commas):")

user_input = st.text_input("Symptoms (e.g. Itching,Skin Rash,Nodal Skin Eruptions)")

if st.button("Predict Disease"):
    input_symptoms = user_input.split(",")
    input_data = [0] * len(symptom_index)
    
    for symptom in input_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])[0]

    st.subheader("Prediction Results")
    st.write(f"🎯 Final Prediction: **{final_pred}**")
    st.write(f"- Random Forest: {rf_pred}")
    st.write(f"- Naive Bayes: {nb_pred}")
    st.write(f"- SVM: {svm_pred}")
