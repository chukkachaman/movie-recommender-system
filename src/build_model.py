"""
Builds the content-based recommendation model:
combines movie metadata into a single "tags" string per movie,
vectorizes it, and computes a cosine-similarity matrix.
Outputs data/movies.pkl and data/similarity.pkl
"""
import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "movies_raw.csv")


def _parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _no_space(items: list[str]) -> list[str]:
    return [item.replace(" ", "").lower() for item in items]


def main():
    df = pd.read_csv(RAW_PATH)
    for col in ("genres", "keywords", "cast", "director"):
        df[col] = df[col].apply(_parse_list)

    df["overview_tokens"] = df["overview"].fillna("").apply(lambda x: x.split())
    df["tags"] = (
        df["overview_tokens"]
        + df["genres"].apply(_no_space)
        + df["keywords"].apply(_no_space)
        + df["cast"].apply(_no_space)
        + df["director"].apply(_no_space)
    )
    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    movies = df[["movie_id", "title", "poster_path", "release_date", "vote_average", "genres"]].copy()

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    with open(os.path.join(DATA_DIR, "movies.pkl"), "wb") as f:
        pickle.dump(movies, f)
    with open(os.path.join(DATA_DIR, "similarity.pkl"), "wb") as f:
        pickle.dump(similarity, f)

    print(f"Built model for {len(movies)} movies. Saved movies.pkl and similarity.pkl")


if __name__ == "__main__":
    main()
