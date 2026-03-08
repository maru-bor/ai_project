import pandas as pd

df = pd.read_json("csfd_movies.jsonl", lines=True)

print(df.head())