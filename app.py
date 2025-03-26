from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model and polynomial transformer
model = joblib.load("model.pkl")
poly = joblib.load("poly_transform.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        time = float(request.form['time'])
        time_poly = poly.transform([[time]])
        predicted_temp = model.predict(time_poly)[0]
        return jsonify({"temperature": round(predicted_temp, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
