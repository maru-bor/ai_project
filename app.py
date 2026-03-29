from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

with open("model.dat", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.dat", "rb") as f:
    vectorizer = pickle.load(f)

with open("mlb.dat", "rb") as f:
    mlb = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\(oficiální text distributora\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_genres = []
    message = ""
    title = ""
    description = ""

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()

        if not title or not description:
            message = "Prosím vyplň název filmu i popis."
        else:
            text = title + " " + clean_text(description)
            X_input = vectorizer.transform([text])
            y_prob = model.predict_proba(X_input)
            threshold = 0.5
            y_pred = (y_prob >= threshold).astype(int)
            predicted_genres = list(mlb.inverse_transform(y_pred)[0])

            if len(predicted_genres) == 0:
                message = "Model nenašel žádný dostatečně pravděpodobný žánr."

    return render_template(
        "index.html",
        predicted_genres=predicted_genres,
        message=message,
        title=title,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=False)