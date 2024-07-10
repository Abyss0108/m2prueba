from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            data = request.form.to_dict()

        # Convertir valores a float
        for key in data:
            data[key] = float(data[key])

        final_features = np.array([data['adjClose'], data['adjHigh'], data['high'], data['low'], data['adjLow']]).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)
        prediction = model.predict(final_features_scaled)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
