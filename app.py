"""
app.py
------
Flask web app that accepts a movie review from the user, runs it
through both a saved RNN model and a saved LSTM model, and displays
each model's prediction plus the accuracy comparison recorded during
training.

Make sure you have already run `python train_models.py` once, so
that rnn_model.h5, lstm_model.h5, word_index.pkl, and accuracy.json
exist in this same folder.

Run with:
    python app.py
Then open 
http://127.0.0.1:5000 in your browser.
"""

import json
import pickle

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200
VOCAB_SIZE = 10000

app = Flask(__name__)

# ---- Load everything once, at startup ----
print("Loading models and word index...")
rnn_model = load_model("rnn_model.h5")
lstm_model = load_model("lstm_model.h5")

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

with open("accuracy.json", "r") as f:
    accuracy = json.load(f)

print("Ready.")


def encode_review(text):
    """Convert raw review text into the same integer-sequence format
    the models were trained on."""
    words = text.lower().split()
    # Keras's IMDB indices are offset by 3 (0,1,2 are reserved tokens)
    encoded = [1]  # 1 = "start of sequence" token
    for word in words:
        idx = word_index.get(word, 2) + 3  # 2 = "unknown word" token
        if idx < VOCAB_SIZE:
            encoded.append(idx)
        else:
            encoded.append(2)
    return pad_sequences([encoded], maxlen=MAX_LEN)


def label_for(score):
    return "Positive" if score >= 0.5 else "Negative"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "")
        if review_text.strip():
            encoded = encode_review(review_text)

            rnn_score = float(rnn_model.predict(encoded, verbose=0)[0][0])
            lstm_score = float(lstm_model.predict(encoded, verbose=0)[0][0])

            result = {
                "rnn_label": label_for(rnn_score),
                "rnn_score": round(rnn_score, 3),
                "lstm_label": label_for(lstm_score),
                "lstm_score": round(lstm_score, 3),
            }

    return render_template(
        "index.html",
        result=result,
        review_text=review_text,
        rnn_accuracy=round(accuracy["rnn_accuracy"] * 100, 2),
        lstm_accuracy=round(accuracy["lstm_accuracy"] * 100, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
