import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("Credit Card Fraud Detection 💳")

# Load the trimmed dataset for display
df_1 = pd.read_csv("creditcard_trimmed.csv")

# Display the dataset
st.header("Credit Card Dataset")
st.write(df_1)

df = pd.read_csv("creditcard.csv")

# Prepare the data
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a scaler and a classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Sidebar inputs for user features
st.header("User Input Features")

def user_input_features():
    amount = st.number_input('Amount', min_value=0.0, max_value=10000.0, value=0.0)
    v1 = st.number_input('V1', min_value=-100.0, max_value=100.0, value=0.0)
    v2 = st.number_input('V2', min_value=-100.0, max_value=100.0, value=0.0)
    v3 = st.number_input('V3', min_value=-100.0, max_value=100.0, value=0.0)
    # Add more inputs for all features as needed

    data = {
        'V1': v1,
        'V2': v2,
        'V3': v3,
        'Amount': amount
    }
    features = pd.DataFrame(data, index=[0])
    return features 

input_df = user_input_features()

# Preprocess the input features
input_df_scaled = pipeline.named_steps['scaler'].transform(input_df)

if st.button("Detect"):
    # Make predictions
    prediction = pipeline.named_steps['classifier'].predict(input_df_scaled)
    prediction_proba = pipeline.named_steps['classifier'].predict_proba(input_df_scaled)
    
    st.subheader('Prediction')
    fraud_status = "**Fraud**" if prediction[0] == 1 else "**Not Fraud**"
    st.write(fraud_status)
