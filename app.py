import streamlit as st
import pickle
import numpy as np

# 1. Trained ML Model ko load karna
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Web App ka Title aur Design
st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="centered")
st.title("🌾 Smart Crop Recommendation System")
st.write("Enter the soil and weather details below to find the best crop to grow.")

# 3. User se Input lene ke liye boxes banana
st.subheader("📊 Soil & Weather Inputs")

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N) value in soil", min_value=0, max_value=200, value=50)
    p = st.number_input("Phosphorus (P) value in soil", min_value=0, max_value=200, value=50)
    k = st.number_input("Potassium (K) value in soil", min_value=0, max_value=200, value=50)
    temp = st.number_input("Temperature (in °C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (in %)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH level of Soil (0-14)", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, max_value=500.0, value=100.0)

# 4. Prediction Button aur Logic
st.write("---")
if st.button("🌱 Predict Best Crop", use_container_width=True):
    # Input data ko array mein convert karna
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
    # Model se predict karwana
    prediction = model.predict(input_data)
    
    # Output ko screen par sundar tarike se dikhana
    st.balloons() # Ek chota sa celebration animation screen par dikhega
    st.success(f"🎉 The best crop recommended for your land is: **{prediction[0].upper()}**")