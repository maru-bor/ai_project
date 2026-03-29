import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\(oficiální text distributora\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_genres(genres):
    return [g for g in genres if g not in genres_to_remove]

df = pd.read_json("csfd_movies.jsonl", lines=True)

df = df.dropna(subset=["description"])
genres_to_remove = ["Katastrofický", "Road movie", "Muzikál", "Western"]
df["genres"] = df["genres"].apply(remove_genres)
df = df[df["genres"].apply(len) > 0]


df["description_clean"] = df["description"].apply(clean_text)
df["text"] = df["title"] + " " + df["description_clean"]


mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["genres"])

vectorizer = TfidfVectorizer(max_features=50000,ngram_range=(1,2),min_df=5,max_df=0.9,sublinear_tf=True)
X = vectorizer.fit_transform(df["text"])

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=11)

model = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Odkomentovat pokud chcete vidět classification report modelu
#print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

