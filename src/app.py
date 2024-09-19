from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained LSTM model
model = tf.keras.models.load_model('models/lstm_sentiment_model.h5')

# Load the tokenizer used for preprocessing
with open('models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Maximum length of input sequence
max_sequence_length = 100

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    review = data['review']
    
    # Preprocess the review
    review_sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(review_sequence, maxlen=max_sequence_length)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    # Return the result as JSON
    return jsonify({'review': review, 'sentiment': sentiment})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)