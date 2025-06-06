
import streamlit as st
import numpy as np
import pickle
from collections import Counter

# Load trained models and other data
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
symptom_index = pickle.load(open('symptom_index.pkl', 'rb'))

# UI
st.title("🧠 Disease Prediction System")
st.write("Select your symptoms from the list below:")

# Convert keys to list for display
available_symptoms = list(symptom_index.keys())
selected_symptoms = st.multiselect("Select Symptoms", available_symptoms)

if st.button("Predict Disease"):
    # Initialize input vector
    input_data = [0] * len(symptom_index)

    # Fill input vector based on selected symptoms
    for symptom in selected_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    st.write("🔍 Input vector to model:", input_data)

    # Predict using all models
    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]

    # Voting
    votes = [rf_pred, nb_pred, svm_pred]
    final_pred = Counter(votes).most_common(1)[0][0]

    # Output
    st.subheader("🩺 Prediction Results")
    st.write(f"🎯 Final Prediction: **{final_pred}**")
    st.write(f"- Random Forest: {rf_pred}")
    st.write(f"- Naive Bayes: {nb_pred}")
    st.write(f"- SVM: {svm_pred}")
