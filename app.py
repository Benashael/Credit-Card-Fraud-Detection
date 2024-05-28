import streamlit as st
import pandas as pd
import joblib  # To load the pre-trained model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("Credit Card Fraud Detection 💳")

# Load the pre-trained model
model = joblib.load('credit_card_fraud_model.pkl')

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Sidebar inputs for user features
st.header("Transaction Details")

def user_input_features():
    amount = st.number_input('Amount', min_value=0.0, max_value=10000.0, value=0.0)
    time = st.number_input('Time', min_value=0, max_value=172800, value=0)
    v1 = st.number_input('V1', min_value=-100.0, max_value=100.0, value=0.0)
    v2 = st.number_input('V2', min_value=-100.0, max_value=100.0, value=0.0)
    v3 = st.number_input('V3', min_value=-100.0, max_value=100.0, value=0.0)
    # Add more inputs for all features as needed

    data = {
        'Amount': amount,
        'Time': time,
        'V1': v1,
        'V2': v2,
        'V3': v3,
        # Add more features as needed
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Preprocess the input features
input_df_scaled = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_df_scaled)
prediction_proba = model.predict_proba(input_df_scaled)

st.subheader('Prediction')
fraud_status = "**Fraud**" if prediction[0] == 1 else "**Not Fraud**"
st.write(fraud_status)

st.subheader('Prediction Probability')
st.write(prediction_proba)
