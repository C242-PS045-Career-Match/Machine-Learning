from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Load model and TF-IDF vectorizer
model = load_model('model.h5')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load job categories from text file
with open('job_categories.txt', 'r') as f:
    job_categories = [line.strip() for line in f.readlines()]

label_mapping = {
    'Design and User Experience': 0,
    'Finance and Business Strategy': 1,
    'Hardware Engineering and Infrastructure': 2,
    'Legal and Communications': 3,
    'Operations and Support': 4,
    'Sales and Marketing': 5,
    'Software Development and IT Services': 6
}

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input texts from request
        input_data = request.json.get('texts', [])

        if not input_data or not isinstance(input_data, list):
            return jsonify({"error": "Invalid input. Provide a list of texts."}), 400

        # Transform input using TF-IDF vectorizer
        processed_input = tfidf_vectorizer.transform(input_data).toarray()

        # Make predictions
        predictions = model.predict(processed_input)

        # Map predictions to categories
        results = []
        for prediction in predictions:
            max_idx = np.argmax(prediction)
            predicted_category = job_categories[max_idx]
            results.append({
                "category": predicted_category,
                "confidence": float(prediction[max_idx])
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
