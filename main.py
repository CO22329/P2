from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# Grade mapping
grade_map = {0: "Normal", 1: "Medium Risk", 2: "High Risk"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        cholesterol = float(request.form['cholesterol'])
        bp = float(request.form['bp'])
        hr = float(request.form['hr'])
        ecg = float(request.form['ecg'])

        # Validate input ranges
        if not (20 <= age <= 100):
            raise ValueError("Age must be between 20 and 100")
        if not (100 <= cholesterol <= 400):
            raise ValueError("Cholesterol must be between 100 and 400 mg/dL")
        if not (80 <= bp <= 200):
            raise ValueError("Blood Pressure must be between 80 and 200 mmHg")
        if not (40 <= hr <= 200):
            raise ValueError("Heart Rate must be between 40 and 200 bpm")
        if not (0 <= ecg <= 2):
            raise ValueError("ECG Result must be between 0 and 2")

        # Scale features and predict
        features = scaler.transform([[age, cholesterol, bp, hr, ecg]])
        prediction = model.predict(features)[0]

        # Create message and recommendation
        risk_level = grade_map.get(prediction, 'Unknown')
        msg = f"Heart Risk Level: <b>{risk_level}</b>"

        recommendations = {
            0: [
                "Excellent! Your heart health is in good condition.",
                "Continue maintaining a healthy lifestyle with regular exercise.",
                "Keep eating a balanced diet rich in fruits and vegetables.",
                "Monitor your health regularly with annual check-ups.",
                "Stay hydrated and get adequate sleep (7-8 hours daily)."
            ],
            1: [
                "Medium risk detected. Take preventive measures now.",
                "Monitor cholesterol and blood pressure regularly.",
                "Increase physical activity to at least 150 minutes per week.",
                "Reduce sodium intake and limit processed foods.",
                "Consider consulting a cardiologist for detailed evaluation.",
                "Quit smoking if applicable and limit alcohol consumption."
            ],
            2: [
                "High risk detected! Immediate action required.",
                "Consult a cardiologist immediately for comprehensive evaluation.",
                "Start medication as prescribed by your doctor.",
                "Follow a strict heart-healthy diet (DASH diet recommended).",
                "Begin supervised exercise program gradually.",
                "Monitor blood pressure and cholesterol weekly.",
                "Consider cardiac stress testing and echocardiogram.",
                "Quit smoking immediately and avoid secondhand smoke."
            ]
        }

        recommendation_list = recommendations.get(prediction, ["Consult a healthcare professional."])

        return render_template('result.html',
                               message=msg,
                               recommendations=recommendation_list,
                               risk_level=risk_level,
                               age=age,
                               cholesterol=cholesterol,
                               bp=bp,
                               hr=hr,
                               ecg=ecg,
                               risk=prediction)

    except ValueError as ve:
        return render_template('index.html', error_message=str(ve))
    except Exception as e:
        return render_template('index.html', error_message=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)