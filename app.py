import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load trained model, encoders, and feature names
@st.cache_resource
def load_model():
    return joblib.load("random_forest_npi_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

@st.cache_resource
def load_feature_names():
    return joblib.load("feature_names.pkl")

@st.cache_data
def load_data():
    """Loads the dataset and pre-processes it."""
    df = pd.read_excel("dummy_npi_data.xlsx", sheet_name="Dataset")
    df["Login Time"] = pd.to_datetime(df["Login Time"])
    df["Logout Time"] = pd.to_datetime(df["Logout Time"])
    df["Login Hour"] = df["Login Time"].dt.hour  # Extract hour
    return df

# Function to predict best doctors at a given time
def predict_doctors(time_input, df, model, encoders, feature_names):
    """Predicts doctors likely to attend survey at a given time."""
    
    # Convert input time to datetime
    input_time = datetime.strptime(time_input, "%H:%M").time()
    
    # Filter doctors who are active during this time
    active_doctors = df[(df["Login Time"].dt.time <= input_time) & 
                        (df["Logout Time"].dt.time >= input_time)]
    
    if active_doctors.empty:
        return pd.DataFrame()  # Return empty if no doctors are available
    
    # Encode categorical features
    for col, encoder in encoders.items():
        active_doctors[col] = encoder.transform(active_doctors[col])

    # Ensure "Login Hour" is included
    active_doctors["Login Hour"] = active_doctors["Login Time"].dt.hour

    # Ensure feature order matches training data
    features = feature_names
    X_input = active_doctors[features]

    # Predict survey likelihood
    active_doctors["Predicted"] = model.predict(X_input)

    # Filter results
    result_df = active_doctors[active_doctors["Predicted"] == 1][["NPI", "Speciality", "Region"]]
    
    return result_df

# Streamlit UI
st.title("📩 Doctor Survey Invitation Predictor")

# Load model, encoders & data
model = load_model()
encoders = load_encoders()
feature_names = load_feature_names()
df = load_data()

# Time input
user_time = st.time_input("⏰ Select a time:")

if st.button("🎯 Get Best Doctors"):
    result_df = predict_doctors(user_time.strftime("%H:%M"), df, model, encoders, feature_names)
    
    if result_df.empty:
        st.warning("⚠️ No doctors available at this time!")
    else:
        st.write(result_df)
        
        # CSV Export
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "recommended_doctors.csv", "text/csv")