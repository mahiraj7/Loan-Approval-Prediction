import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")

st.title("Loan Approval Prediction")

gender = st.selectbox("Gender", [0,1])
married = st.selectbox("Married", [0,1])
income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.selectbox("Credit History", [0,1])

input_data = np.array([
    gender,
    married,
    0,
    1,
    0,
    income,
    0,
    loan_amount,
    360,
    credit_history,
    1
]).reshape(1,-1)

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
