import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained encoders, scaler, and model
try:
    with open('model_and_encoders.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    voice_plan_encoder = loaded_data['voice_plan_encoder']
    intl_plan_encoder = loaded_data['intl_plan_encoder']
    churn_encoder = loaded_data['churn_encoder']
    scaler = loaded_data['scaler']
    model = loaded_data['model']
except (FileNotFoundError, KeyError) as e:
    st.error("Error loading model or encoders. Please check the file path or structure.")
    st.stop()

# Streamlit UI
st.title("Customer Churn Prediction")

# Create input fields for each feature
account_length = st.number_input("Account Length", min_value=0)
voice_plan = st.selectbox("Voice Plan", ["yes", "no"])
intl_plan = st.selectbox("International Plan", ["yes", "no"])
intl_mins = st.number_input("International Minutes", min_value=0.0)
intl_calls = st.number_input("International Calls", min_value=0)
day_mins = st.number_input("Day Minutes", min_value=0.0)
day_calls = st.number_input("Day Calls", min_value=0)
eve_mins = st.number_input("Evening Minutes", min_value=0.0)
eve_calls = st.number_input("Evening Calls", min_value=0)
night_mins = st.number_input("Night Minutes", min_value=0.0)
night_calls = st.number_input("Night Calls", min_value=0)
customer_calls = st.number_input("Customer Service Calls", min_value=0)

# Prepare input data
try:
    input_data = pd.DataFrame({
        'account.length': [account_length],
        'voice.plan': [voice_plan_encoder.transform([voice_plan])[0]],
        'intl.plan': [intl_plan_encoder.transform([intl_plan])[0]],
        'intl.mins': [intl_mins],
        'intl.calls': [intl_calls],
        'day.mins': [day_mins],
        'day.calls': [day_calls],
        'eve.mins': [eve_mins],
        'eve.calls': [eve_calls],
        'night.mins': [night_mins],
        'night.calls': [night_calls],
        'customer.calls': [customer_calls]
    })

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Reshape to add a third dimension for compatibility
    input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

    # Predict churn when the button is clicked
    if st.button("Predict Churn"):
        prediction = model.predict(input_data_scaled)
        st.write("Predicted: ", prediction)
        if prediction >= 0.5: 
            st.write("Predicted Churn: Yes.")
        else : 
            st.write("Predicted Churn: No.")
            

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
