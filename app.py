from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = r'C:\Users\KIIT\Desktop\render-demo\model .pkl'

try:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model is not loaded. Cannot make predictions.")
    
    try:
        # Extract features from form
        cgpa = float(request.form['cgpa'])
        iq = int(request.form['iq'])
        profile_score = int(request.form['profile_score'])

        # Make prediction
        final_features = np.array([[cgpa, iq, profile_score]])
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
