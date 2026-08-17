# CineMatch — Movie Recommender System

A content-based movie recommender built with Python, scikit-learn, and Streamlit.
Pick a movie you like and get similar picks based on plot overview, genres,
keywords, cast, and director.

## How it works

1. **`src/fetch_data.py`** — pulls popular movies from the [TMDB API](https://www.themoviedb.org/documentation/api)
   (title, overview, genres, keywords, top cast, director, poster path, rating)
   and writes them to `data/movies_raw.csv`.
2. **`src/build_model.py`** — combines each movie's metadata into a single
   "tags" string, vectorizes it with `CountVectorizer`, and computes a
   cosine-similarity matrix across all movies. Outputs `data/movies.pkl` and
   `data/similarity.pkl`.
3. **`app.py`** — a Streamlit UI where you search for a movie and get the
   6 most similar titles, with posters, release year, and rating.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your TMDB_API_KEY
```

Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api).

## Build the dataset and model

```bash
python src/fetch_data.py
python src/build_model.py
```

## Run the app

```bash
streamlit run app.py
```

## Project structure

```
.
├── app.py                 # Streamlit UI
├── src/
│   ├── fetch_data.py      # TMDB data collection
│   └── build_model.py     # tag vectorization + similarity matrix
├── data/                  # dataset + model artifacts (raw CSV gitignored)
├── .streamlit/config.toml # dark theme config
└── requirements.txt
```
