import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models and scaler
rf_model = joblib.load("churn_model_rf.pkl")
xgb_model = joblib.load("churn_model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Telecommunications Customer Churn Prediction")

# Sidebar for user input
st.sidebar.header("Customer Information")

def user_input_features():
    account_length = st.sidebar.number_input("Account Length", min_value=0, value=100)
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("Yes", "No"))
    voice_mail_messages = st.sidebar.number_input("Voice Mail Messages", min_value=0, value=0)
    day_mins = st.sidebar.number_input("Day Minutes", min_value=0.0, value=200.0)
    day_calls = st.sidebar.number_input("Day Calls", min_value=0, value=100)
    day_charge = st.sidebar.number_input("Day Charge", min_value=0.0, value=50.0)
    evening_mins = st.sidebar.number_input("Evening Minutes", min_value=0.0, value=200.0)
    evening_calls = st.sidebar.number_input("Evening Calls", min_value=0, value=100)
    evening_charge = st.sidebar.number_input("Evening Charge", min_value=0.0, value=50.0)
    night_mins = st.sidebar.number_input("Night Minutes", min_value=0.0, value=200.0)
    night_calls = st.sidebar.number_input("Night Calls", min_value=0, value=100)
    night_charge = st.sidebar.number_input("Night Charge", min_value=0.0, value=50.0)
    international_plan = st.sidebar.selectbox("International Plan", ("Yes", "No"))
    international_mins = st.sidebar.number_input("International Minutes", min_value=0.0, value=10.0)
    international_calls = st.sidebar.number_input("International Calls", min_value=0, value=5)
    international_charge = st.sidebar.number_input("International Charge", min_value=0.0, value=5.0)
    customer_service_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, value=1)

    # Map yes/no to 1/0
    voice_mail_plan = 1 if voice_mail_plan.lower() == "yes" else 0
    international_plan = 1 if international_plan.lower() == "yes" else 0

    # Create total mins and charges (feature engineering)
    total_mins = day_mins + evening_mins + night_mins + international_mins
    total_charge = day_charge + evening_charge + night_charge + international_charge
    avg_day_call = day_mins/day_calls if day_calls != 0 else 0
    avg_night_call = night_mins/night_calls if night_calls != 0 else 0
    high_customer_service_calls = 1 if customer_service_calls > 3 else 0

    data = {
        "account_length": account_length,
        "voice_mail_plan": voice_mail_plan,
        "voice_mail_messages": voice_mail_messages,
        "day_mins": day_mins,
        "day_calls": day_calls,
        "day_charge": day_charge,
        "evening_mins": evening_mins,
        "evening_calls": evening_calls,
        "evening_charge": evening_charge,
        "night_mins": night_mins,
        "night_calls": night_calls,
        "night_charge": night_charge,
        "international_plan": international_plan,
        "international_mins": international_mins,
        "international_calls": international_calls,
        "international_charge": international_charge,
        "customer_service_calls": customer_service_calls,
        "total_mins": total_mins,
        "total_charge": total_charge,
        "avg_day_call": avg_day_call,
        "avg_night_call": avg_night_call,
        "high_customer_service_calls": high_customer_service_calls
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Feature scaling (optional)
scaled_input = scaler.transform(input_df)

# Prediction buttons
st.subheader("Predictions")
if st.button("Predict Churn with Random Forest"):
    prediction = rf_model.predict(scaled_input)[0]
    prediction_prob = rf_model.predict_proba(scaled_input)[0][1]
    st.write(f"Prediction: {'Churn' if prediction==1 else 'Loyal'}")
    st.write(f"Churn Probability: {prediction_prob:.2f}")

if st.button("Predict Churn with XGBoost"):
    prediction = xgb_model.predict(scaled_input)[0]
    prediction_prob = xgb_model.predict_proba(scaled_input)[0][1]
    st.write(f"Prediction: {'Churn' if prediction==1 else 'Loyal'}")
    st.write(f"Churn Probability: {prediction_prob:.2f}")
