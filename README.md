# 🎬 Movie Recommendation System

A beginner-friendly machine learning project that recommends movies with similar plot descriptions.

## Features
- Select a movie from a dropdown.
- Get up to five similar movie recommendations.
- Uses content-based filtering.
- Simple Streamlit web interface.
- Includes a small sample dataset so you can run it immediately.

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Cosine Similarity

## Project files
```text
Movie-Recommendation-System/
├── app.py
├── movies.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup in VS Code

1. Install Python and VS Code.
2. Open this folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Recommended) Create and activate a virtual environment:

   Windows:
   ```bash
   python -m venv .venv
   python -m venv .venv
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```

Streamlit will show a local URL in the terminal. Open it in your browser.

## Dataset

The included `movies.csv` is a small sample dataset created for learning and testing. It lets you run the project immediately.

To expand the project, you can use the TMDB 5000 Movie Dataset:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

If you use that dataset, prepare a CSV named `movies.csv` with these two columns:
- `title`
- `overview`

Check the dataset's licence and redistribution terms before publishing a downloaded dataset to a public repository. Do not upload data unless its licence permits redistribution.

## How it works

1. **TF-IDF** converts movie descriptions into numerical feature vectors.
2. **Cosine similarity** compares those vectors.
3. The app ranks movies by similarity and displays the top five, excluding the selected movie.

This is a **content-based recommendation system**. It recommends based on descriptions, not personal viewing history.

## Ideas for improvements
- Add movie posters using a permitted movie API.
- Add genre and rating filters.
- Add a search box.
- Deploy the app online.

## Author
Elanchezhiyan V
