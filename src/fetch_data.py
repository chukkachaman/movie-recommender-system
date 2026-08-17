"""
Collects movie metadata from the TMDB API (popular movies + full details)
and writes it to data/movies_raw.csv.
"""
import os
import time
import concurrent.futures as cf

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
NUM_PAGES = 200  # ~20 movies/page -> ~4000 movies
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "movies_raw.csv")


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=20)
    session.mount("https://", adapter)
    return session


SESSION = make_session()


def get_popular_movie_ids(num_pages: int) -> list[int]:
    ids = []
    for page in tqdm(range(1, num_pages + 1), desc="Discovering movies"):
        resp = SESSION.get(
            f"{BASE_URL}/discover/movie",
            params={
                "api_key": API_KEY,
                "sort_by": "popularity.desc",
                "page": page,
                "include_adult": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        ids.extend(m["id"] for m in results)
    return ids


def fetch_movie_details(movie_id: int) -> dict | None:
    try:
        resp = SESSION.get(
            f"{BASE_URL}/movie/{movie_id}",
            params={
                "api_key": API_KEY,
                "append_to_response": "credits,keywords",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()

        genres = [g["name"] for g in data.get("genres", [])]
        keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
        cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
        crew = data.get("credits", {}).get("crew", [])
        director = [c["name"] for c in crew if c.get("job") == "Director"][:1]

        return {
            "movie_id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview") or "",
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "poster_path": data.get("poster_path"),
            "release_date": data.get("release_date"),
            "vote_average": data.get("vote_average"),
        }
    except requests.RequestException:
        return None


def main():
    if not API_KEY:
        raise SystemExit("TMDB_API_KEY not set. Add it to your .env file.")

    movie_ids = get_popular_movie_ids(NUM_PAGES)
    movie_ids = list(dict.fromkeys(movie_ids))  # de-dupe, keep order
    print(f"Fetching details for {len(movie_ids)} movies...")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_movie_details, mid): mid for mid in movie_ids}
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Fetching details"):
            result = future.result()
            if result and result["overview"]:
                rows.append(result)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} movies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
