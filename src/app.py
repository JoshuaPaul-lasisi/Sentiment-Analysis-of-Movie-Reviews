from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('models/lstm_sentiment_model.keras')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    # Tokenize and pad the input
    seq = tokenizer.texts_to_sequences([data])
    padded_seq = pad_sequences(seq, maxlen=200)

    # Make a prediction
    prediction = (model.predict(padded_seq) > 0.5).astype("int32")

    # Return the result
    return jsonify({'sentiment': int(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)