from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the saved LSTM model
model = tf.keras.models.load_model(os.path.join('models', 'lstm_sentiment_model.keras'))

# Load tokenizer (if you saved it during training)
# If not available, you'll have to re-create it using the same logic you used during training
try:
    with open('models/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
except FileNotFoundError:
    # Re-create tokenizer if it wasn't saved
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=5000)  # Change this to match training
    tokenizer.fit_on_texts([])  # Dummy fit; replace with real training data if needed

# Preprocess the review text (same steps as in your notebook)
def preprocess_review(review):
    review = review.lower()  # Lowercase
    review = re.sub('<.*?>', '', review)  # Remove HTML tags
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove special characters
    tokens = word_tokenize(review)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    
    # Preprocess the input text
    review = preprocess_review(data['review'])
    
    # Tokenize and pad the review text
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=200)
    
    # Make prediction
    prediction = model.predict(review_pad)
    
    # Convert the prediction to a label
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)