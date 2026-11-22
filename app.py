import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

@st.cache_data
def load_data():
    data = pd.read_csv("movies.csv")

    required_columns = {"title", "overview"}
    if not required_columns.issubset(data.columns):
        raise ValueError("movies.csv must contain 'title' and 'overview' columns.")

    data = data[["title", "overview"]].dropna(subset=["title"])
    data["overview"] = data["overview"].fillna("").astype(str)
    data = data.drop_duplicates(subset=["title"])
    return data.reset_index(drop=True)

@st.cache_resource
def create_model(descriptions):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(descriptions)
    return cosine_similarity(matrix)

try:
    movies = load_data()

    if movies.empty:
        st.error("The movie dataset is empty. Add movie records to movies.csv.")
        st.stop()

    similarity = create_model(tuple(movies["overview"].tolist()))

    def recommend(movie_name):
        matches = movies[
            movies["title"].str.casefold() == movie_name.casefold()
        ]

        if matches.empty:
            return []

        movie_index = matches.index[0]
        scores = list(enumerate(similarity[movie_index]))
        scores.sort(key=lambda item: item[1], reverse=True)

        recommendations = []
        for index, score in scores:
            if index == movie_index:
                continue
            recommendations.append({
                "Movie": movies.iloc[index]["title"],
                "Similarity": round(float(score), 3)
            })
            if len(recommendations) == 5:
                break

        return recommendations

    selected_movie = st.selectbox(
        "Choose your favourite movie:",
        movies["title"].tolist()
    )

    if st.button("Recommend Movies", type="primary"):
        results = recommend(selected_movie)

        if results:
            st.subheader("🍿 Recommended Movies")
            for number, movie in enumerate(results, start=1):
                st.write(f"{number}. **{movie['Movie']}**")
                st.caption(f"Description similarity score: {movie['Similarity']}")
        else:
            st.info("No similar movies found. Try another title.")

    st.caption(f"Loaded {len(movies)} movies.")

except FileNotFoundError:
    st.error("Could not find movies.csv. Place it in the same folder as app.py.")
except Exception as error:
    st.error(f"Something went wrong: {error}")
