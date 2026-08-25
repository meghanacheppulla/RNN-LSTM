"""
train_models.py
----------------
Trains an RNN model and an LSTM model on the IMDB movie review
sentiment dataset (built into Keras — no manual download needed),
saves both models to disk, saves the word index used for encoding
reviews, and saves each model's test accuracy so the Flask app can
display a comparison.

Run this ONCE before starting the Flask app:
    python train_models.py
"""

import json
import pickle

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# ---- Settings ----
VOCAB_SIZE = 10000      # only keep the top 10,000 most common words
MAX_LEN = 200            # pad/truncate every review to 200 words
EMBED_DIM = 32
EPOCHS = 3                # keep low so training finishes quickly
BATCH_SIZE = 128

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print("Padding sequences...")
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)


def build_rnn_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        SimpleRNN(32),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


print("\n=== Training RNN model ===")
rnn_model = build_rnn_model()
rnn_model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
)
rnn_loss, rnn_acc = rnn_model.evaluate(x_test, y_test, verbose=0)
print(f"RNN test accuracy: {rnn_acc:.4f}")

print("\n=== Training LSTM model ===")
lstm_model = build_lstm_model()
lstm_model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
)
lstm_loss, lstm_acc = lstm_model.evaluate(x_test, y_test, verbose=0)
print(f"LSTM test accuracy: {lstm_acc:.4f}")

# ---- Save everything the Flask app needs ----
print("\nSaving models...")
rnn_model.save("rnn_model.h5")
lstm_model.save("lstm_model.h5")

print("Saving word index...")
word_index = imdb.get_word_index()
with open("word_index.pkl", "wb") as f:
    pickle.dump(word_index, f)

print("Saving accuracy results...")
with open("accuracy.json", "w") as f:
    json.dump({"rnn_accuracy": float(rnn_acc), "lstm_accuracy": float(lstm_acc)}, f)

print("\nDone! Files created: rnn_model.h5, lstm_model.h5, word_index.pkl, accuracy.json")
print("You can now run: python app.py")
