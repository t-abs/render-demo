import streamlit as st
import numpy as np
import pickle
import os

# Load the model
model_path = r'C:\Users\KIIT\Desktop\render-demo\model .pkl'

try:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# Streamlit App UI
st.title("🎓 Placement Prediction App")

if model_loaded:
    st.markdown("#### Please fill in your details below:")

    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
    iq = st.number_input("IQ", min_value=0, max_value=200, step=1)
    profile_score = st.slider("Profile Score", 0, 100)

    if st.button("Predict Placement"):
        try:
            input_features = np.array([[cgpa, iq, profile_score]])
            prediction = model.predict(input_features)
            output = '🎉 Placed' if prediction[0] == 1 else '❌ Not Placed'
            st.success(f"Prediction: **{output}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("⚠️ Model not loaded. Cannot make predictions.")
