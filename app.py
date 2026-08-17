import os
import pickle

import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1c1f2b/8a8fa3?text=No+Poster"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    with open(os.path.join(DATA_DIR, "movies.pkl"), "rb") as f:
        movies = pickle.load(f)
    with open(os.path.join(DATA_DIR, "similarity.pkl"), "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


def poster_url(poster_path):
    if isinstance(poster_path, str) and poster_path:
        return f"{POSTER_BASE}{poster_path}"
    return PLACEHOLDER_POSTER


def recommend(title: str, movies: pd.DataFrame, similarity, top_n: int = 6):
    matches = movies.index[movies["title"] == title]
    if len(matches) == 0:
        return []
    idx = matches[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    return [movies.iloc[i] for i, _ in scores]


CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #1a1d2e 0%, #0f1117 60%);
}
h1.cine-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4d6d, #ff8fa3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
p.cine-subtitle {
    color: #8a8fa3;
    font-size: 1.05rem;
    margin-top: 0.2rem;
    margin-bottom: 1.8rem;
}
.movie-card {
    background: #171a26;
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid #262a3d;
}
.movie-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 24px rgba(255, 77, 109, 0.25);
    border-color: #ff4d6d;
}
.movie-card img {
    width: 100%;
    display: block;
    aspect-ratio: 2 / 3;
    object-fit: cover;
}
.movie-info {
    padding: 0.7rem 0.8rem 1rem 0.8rem;
}
.movie-info .title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #f2f2f5;
    line-height: 1.25rem;
    height: 2.5rem;
    overflow: hidden;
}
.movie-info .meta {
    color: #8a8fa3;
    font-size: 0.8rem;
    margin-top: 0.3rem;
}
.rating-badge {
    display: inline-block;
    background: #ff4d6d;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<h1 class="cine-title">🎬 CineMatch</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="cine-subtitle">Pick a movie you love — get similar picks based on plot, genre, cast and crew.</p>',
    unsafe_allow_html=True,
)

try:
    movies_df, similarity_matrix = load_data()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python src/fetch_data.py` then "
        "`python src/build_model.py` first to build the dataset."
    )
    st.stop()

col1, col2 = st.columns([4, 1])
with col1:
    selected_title = st.selectbox(
        "Search for a movie",
        options=movies_df["title"].sort_values().tolist(),
        index=None,
        placeholder="Start typing a movie title...",
    )
with col2:
    st.write("")
    st.write("")
    go = st.button("Recommend", type="primary", use_container_width=True)

if go and selected_title:
    recs = recommend(selected_title, movies_df, similarity_matrix, top_n=6)

    if not recs:
        st.warning("No recommendations found for that title.")
    else:
        st.subheader(f"Because you liked {selected_title}")
        cols = st.columns(6)
        for col, movie in zip(cols, recs):
            year = str(movie["release_date"])[:4] if pd.notna(movie["release_date"]) else "—"
            rating = f"{movie['vote_average']:.1f}" if pd.notna(movie["vote_average"]) else "N/A"
            with col:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{poster_url(movie['poster_path'])}" />
                        <div class="movie-info">
                            <div class="title">{movie['title']}</div>
                            <div class="meta">{year} &nbsp;•&nbsp; <span class="rating-badge">★ {rating}</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
elif go and not selected_title:
    st.info("Pick a movie first.")
