"""
app.py
------
Flask web app that accepts a movie review from the user, runs it
through both a saved RNN model and a saved LSTM model, and displays
each model's prediction plus the accuracy comparison recorded during
training.

Now includes user authentication (signup/login/logout) using EMAIL as
the login identifier — only logged-in users can access the sentiment
analysis page. The page greets the user with a display name derived
from their email (the part before "@").

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

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from flask_bcrypt import Bcrypt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import db, User

MAX_LEN = 200
VOCAB_SIZE = 10000

app = Flask(__name__)

# ---- Auth / database configuration ----
app.config["SECRET_KEY"] = "change-this-to-a-random-secret-string"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

db.init_app(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---- Load ML models once, at startup ----
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


# ---- Auth routes (email-based) ----
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists.", "error")
            return redirect(url_for("signup"))

        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(email=email, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully!", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---- Main sentiment analysis route (protected) ----
@app.route("/", methods=["GET", "POST"])
@login_required
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

    # Derive a friendly display name from the email (part before "@")
    display_name = current_user.email.split("@")[0]

    return render_template(
        "index.html",
        result=result,
        review_text=review_text,
        rnn_accuracy=round(accuracy["rnn_accuracy"] * 100, 2),
        lstm_accuracy=round(accuracy["lstm_accuracy"] * 100, 2),
        display_name=display_name,
    )


if __name__ == "__main__":
    app.run(debug=True)