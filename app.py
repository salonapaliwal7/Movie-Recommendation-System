import streamlit as st
import joblib

from src.utils import load_data, preprocess_movies
from src.hybrid import hybrid_recommend


import os

if not os.path.exists("models/cosine_similarity.pkl"):
    from train import train_models 
    train_models()

cosine_sim = joblib.load("models/cosine_similarity.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# LOAD DATA
movies, ratings, tags, links = load_data()
movies = preprocess_movies(movies)

cosine_sim = joblib.load("models/cosine_similarity.pkl")
svd_model = joblib.load("models/svd_model.pkl")


# TITLE
st.title("🎬 Hybrid Movie Recommendation System")

st.markdown("""
A Movie Recommendation System built using

- 🎯 Content-Based Filtering (TF-IDF + Cosine Similarity)
- ⭐ Collaborative Filtering (SVD)
- 🔥 Hybrid Recommendation Engine
""")

st.markdown("---")


# METRICS
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎥 Movies", len(movies))

with col2:
    st.metric("👥 Users", ratings["userId"].nunique())

with col3:
    st.metric("⭐ Ratings", len(ratings))

st.markdown("---")


# SIDEBAR
st.sidebar.header("Recommendation Settings")

movie_name = st.sidebar.selectbox(
    "🎥 Select a Movie",
    sorted(movies["title"].unique())
)

user_id = st.sidebar.number_input(
    "👤 User ID",
    min_value=1,
    value=1,
    step=1
)

top_n = st.sidebar.slider(
    "📊 Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

recommend = st.sidebar.button("🚀 Recommend")


# RECOMMENDATIONS
if recommend:

    with st.spinner("Generating recommendations..."):

        recommendations = hybrid_recommend(
            user_id=user_id,
            movie_name=movie_name,
            movies=movies,
            ratings=ratings,
            cosine_sim=cosine_sim,
            model=svd_model
        )

    st.success("Recommendations generated successfully!")

    st.subheader("✨ Top Movie Recommendations")

    for i, row in enumerate(recommendations.head(top_n).itertuples(), start=1):

        with st.container():

            st.markdown(f"### {i}. 🎬 {row.title}")

            st.write(f"**Genres:** {row.genres}")

            st.divider()


# DATASET INFO
with st.expander("📚 Dataset Information"):

    st.write(f"Movies : {len(movies)}")

    st.write(f"Ratings : {len(ratings)}")

    st.write(f"Users : {ratings['userId'].nunique()}")

    st.write(f"Dataset : MovieLens Latest Small")


# FOOTER
st.markdown("---")

st.caption(
    "Developed using Python • Scikit-Learn • Scikit-Surprise • Streamlit • MovieLens Dataset"
)