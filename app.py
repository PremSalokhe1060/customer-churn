# Gender -> 1 Female  0 Male
# Churn  -> 1 Yes   0 No
# Scaler is exported as scaler.pkl
# model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Predication App")

st.divider()

st.write("Please enter the value and hit predict button for getting a predication.")

st.divider()

# Take Inputs from user 
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

gender = st.selectbox("Enter Gender", ["Male", "Female"])

st.divider()

predictebutton = st.button("Predict")

st.divider()

if predictebutton:

    gender_selected = 1 if gender=="Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction==1 else "No"

    st.balloons()

    st.write(f"Predicated : {predicted}")



else:
    st.write("Please enter the values for predication")