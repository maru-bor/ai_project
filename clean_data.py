import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

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




