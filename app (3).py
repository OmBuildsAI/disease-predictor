
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from collections import Counter
import pickle
import os

def train_and_save_models():
    st.write("### Upload your disease dataset CSV file for training")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        
        if 'disease' not in data.columns:
            st.error("Dataset must contain a 'disease' column for labels.")
            return
        
        encoder = LabelEncoder()
        data['disease'] = encoder.fit_transform(data['disease'])
        
        symptom_cols = data.columns.drop('disease')
        symptom_index = {symptom: idx for idx, symptom in enumerate(symptom_cols)}
        
        X = data[symptom_cols]
        y = data['disease']
        
        st.write("Training models, please wait...")
        
        rf = RandomForestClassifier(random_state=42)
        nb = GaussianNB()
        svm = SVC(probability=True, random_state=42)
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        
        rf.fit(X, y)
        nb.fit(X, y)
        svm.fit(X, y)
        xgb_model.fit(X, y)
        
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(rf, f)
        with open('nb_model.pkl', 'wb') as f:
            pickle.dump(nb, f)
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump(svm, f)
        with open('xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        with open('encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('symptom_index.pkl', 'wb') as f:
            pickle.dump(symptom_index, f)
        
        st.success("✅ Models trained and saved! Please rerun the app.")

# Check if models exist
required_files = ['rf_model.pkl', 'nb_model.pkl', 'svm_model.pkl', 'xgb_model.pkl', 'encoder.pkl', 'symptom_index.pkl']

if not all(os.path.exists(file) for file in required_files):
    st.warning("Models not found. Please upload dataset to train models.")
    train_and_save_models()
else:
    # Load models
    rf_model = pickle.load(open('rf_model.pkl', 'rb'))
    nb_model = pickle.load(open('nb_model.pkl', 'rb'))
    svm_model = pickle.load(open('svm_model.pkl', 'rb'))
    xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    symptom_index = pickle.load(open('symptom_index.pkl', 'rb'))
    
    st.title("Disease Prediction System")
    st.write("Enter your symptoms separated by commas (e.g. Itching,Skin Rash,Nodal Skin Eruptions)")
    
    user_input = st.text_input("Symptoms:")
    
    if st.button("Predict Disease"):
        input_symptoms = [sym.strip() for sym in user_input.split(",")]
        input_data = [0] * len(symptom_index)
        
        for symptom in input_symptoms:
            if symptom in symptom_index:
                input_data[symptom_index[symptom]] = 1
            else:
                st.warning(f"Symptom '{symptom}' not recognized.")
        
        input_data = np.array(input_data).reshape(1, -1)
        
        rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
        nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
        svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]
        xgb_pred = encoder.classes_[xgb_model.predict(input_data)[0]]
        
        votes = [rf_pred, nb_pred, svm_pred, xgb_pred]
        final_pred = Counter(votes).most_common(1)[0][0]
        
        st.subheader("Prediction Results")
        st.write(f"🎯 Final Prediction: *{final_pred}*")
        st.write(f"- Random Forest: {rf_pred}")
        st.write(f"- Naive Bayes: {nb_pred}")
        st.write(f"- SVM: {svm_pred}")
        st.write(f"- XGBoost: {xgb_pred}")
