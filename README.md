# Customer-Churn-Prediction

Customer Churn Prediction on Telecom Dataset

# Telecom Customer Churn Prediction

Overview

This project focuses on predicting customer churn in the telecom industry. The application uses machine learning to identify customers likely to churn, helping telecom companies take proactive measures to retain them.

Features

Predict churn probability based on customer data.

User-friendly interface built with Streamlit.

Scalable preprocessing with feature encoders and scaling techniques.

Model trained and saved for real-time predictions.

Dataset

The dataset used includes customer-specific features such as:

● state: Categorical, for the 51 states and the District of Columbia.

● Area.code

● account.length: how long the account has Business Objective: Customer churn is a big problem for telecommunications companies. Indeed, their annual churn rates are usually higher than 10%. For that reason, they develop strategies to keep as many clients as possible. This is a classification project since the variable to be predicted is binary (churn or loyal customer). The goal here is to model churn probability, conditioned on the customer features. telecommunications

● been active.

● voice.plan: yes or no, voicemail plan.

● voice.messages: number of voicemail messages.

● intl.plan: yes or no, international plan.

● intl.mins: minutes customer used service to make international calls.

● intl.calls: total number of international calls.

● intl.charge: total international charge.

● day.mins: minutes customer used service during the day.

● day.calls: total number of calls during the day.

● day.charge: total charge during the day.

● eve.mins: minutes customer used service during the evening.

● eve.calls: total number of calls during the evening.

● eve.charge: total charge during the evening.

● night.mins: minutes customer used service during the night.

● night.calls: total number of calls during the night.

● night.charge: total charge during the night.

● customer.calls: number of calls to customer service.

● churn: Categorical, yes or no. Indicator of whether the customer has left the company (yes or no).

Technologies Used

Python for data processing and modeling.

Streamlit for building the web interface.

Pickle for loading trained models and encoders.

NumPy and Pandas for data manipulation.

Machine Learning Model: Pretrained and loaded for real-time predictions.

Application Workflow

Input: Users provide customer data via the web interface.

Processing:

Encodes categorical features.

Scales numeric features using a trained scaler.

Prediction:

Runs the data through the model to predict churn probability.

Displays whether the customer is likely to churn.

