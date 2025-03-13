import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify

app = Flask(__name__)

# Check if the model file exists
import os
model_file = "digital_twin_model.h5"

if os.path.exists(model_file):
    print("📢 Loading existing model...")
    model = tf.keras.models.load_model(model_file)
else:
    print("⚠️ Model file not found! Training a new model...")

    # Simulate historical data
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.rand(100)
    })

    X = data[['feature1', 'feature2']]
    y = data['target']

    # Train a simple neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=1)

    # Save the trained model
    model.save(model_file)
    print("✅ Model trained and saved as", model_file)

@app.route('/predict', methods=['GET'])
def predict():
    prediction = model.predict(np.array([[0.5, 0.2]]))  # Example input
    return jsonify({'predicted_value': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(port=5000)
