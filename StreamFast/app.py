import streamlit as st
import pandas as pd
import requests
import time

st.title("Bank Churn Prediction")

# ---------------------------
# Input form
# ---------------------------
with st.form("customer_form"):
    Customer_Age = st.number_input("Customer Age", min_value=18, max_value=100, value=40)
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Dependent_count = st.number_input("Dependent Count", min_value=0, max_value=10, value=2)
    Education_Level = st.selectbox("Education Level", ["High School", "Graduate", "Uneducated", "Unknown", "College", "Post-Graduate", "Doctorate"])
    Marital_Status = st.selectbox("Marital Status", ["Married", "Single", "Unknown", "Divorced"])
    Income_Category = st.selectbox("Income Category", ["$60K - $80K", "Less than $40K", "$80K - $120K", "$40K - $60K", "$120K +", "Unknown"])
    Card_Category = st.selectbox("Card Category", ["Blue", "Gold", "Silver", "Platinum"])
    Months_on_book = st.number_input("Months on Book", min_value=1, max_value=100, value=36)
    Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=4)
    Months_Inactive_12_mon = st.number_input("Months Inactive (12 mon)", min_value=0, max_value=12, value=1)
    Contacts_Count_12_mon = st.number_input("Contacts Count (12 mon)", min_value=0, max_value=10, value=2)
    Credit_Limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0)
    Total_Revolving_Bal = st.number_input("Total Revolving Balance", min_value=0.0, value=500.0)
    Avg_Open_To_Buy = st.number_input("Avg Open To Buy", min_value=0.0, value=4500.0)
    Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amt Chng Q4/Q1", min_value=0.0, value=1.2)
    Total_Trans_Amt = st.number_input("Total Transaction Amt", min_value=0, value=1000)
    Total_Trans_Ct = st.number_input("Total Transaction Count", min_value=0, value=20)
    Total_Ct_Chng_Q4_Q1 = st.number_input("Total Ct Chng Q4/Q1", min_value=0.0, value=1.1)
    Avg_Utilization_Ratio = st.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.2)
    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction request
# ---------------------------
if submitted:
    payload = [{
        "Customer_Age": Customer_Age,
        "Gender": 1 if Gender=="Male" else 0,
        "Dependent_count": Dependent_count,
        "Education_Level": Education_Level,
        "Marital_Status": Marital_Status,
        "Income_Category": Income_Category,
        "Card_Category": Card_Category,
        "Months_on_book": Months_on_book,
        "Total_Relationship_Count": Total_Relationship_Count,
        "Months_Inactive_12_mon": Months_Inactive_12_mon,
        "Contacts_Count_12_mon": Contacts_Count_12_mon,
        "Credit_Limit": Credit_Limit,
        "Total_Revolving_Bal": Total_Revolving_Bal,
        "Avg_Open_To_Buy": Avg_Open_To_Buy,
        "Total_Amt_Chng_Q4_Q1": Total_Amt_Chng_Q4_Q1,
        "Total_Trans_Amt": Total_Trans_Amt,
        "Total_Trans_Ct": Total_Trans_Ct,
        "Total_Ct_Chng_Q4_Q1": Total_Ct_Chng_Q4_Q1,
        "Avg_Utilization_Ratio": Avg_Utilization_Ratio
    }]

    # Call FastAPI
    API_URL = "http://localhost:8000/predict"  # safer than 127.0.0.1

    # Retry if FastAPI isn't ready yet
    for _ in range(10):
        try:
            response = requests.post(API_URL, json=payload)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(2)
    else:
        st.error("Failed to connect to FastAPI. Please try again later.")
        st.stop()

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Class: {result[0]['Predicted_Class']}, Churn Probability: {result[0]['Churn_Probability']:.2f}")
    else:
        st.error(f"API Error: {response.status_code}")

