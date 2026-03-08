import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_json("csfd_movies.jsonl", lines=True)

df = df.dropna(subset=["description"])
df = df[df["genres"].apply(len) > 0]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\(oficiální text distributora\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["description_clean"] = df["description"].apply(clean_text)
df["text"] = df["title"] + " " + df["description_clean"]

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["genres"])

vectorizer = TfidfVectorizer(max_features=50000,ngram_range=(1,2),min_df=5,max_df=0.9)
X = vectorizer.fit_transform(df["text"])

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=11)


