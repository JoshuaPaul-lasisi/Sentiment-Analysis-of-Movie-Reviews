# src/models/load_model.py
import tensorflow as tf

def load_lstm_model(model_path):
    """Loads and returns the pre-trained LSTM model."""
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

if __name__ == '__main__':
    model_path = 'models/lstm_sentiment_model.keras'
    model = load_lstm_model(model_path)