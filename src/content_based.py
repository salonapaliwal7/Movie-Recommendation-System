from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import joblib


def train_content_model(movies):

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(movies["features"])

    cosine_sim = cosine_similarity(tfidf_matrix)

    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump(cosine_sim, "models/cosine_similarity.pkl")

    return cosine_sim

def recommend_movies(title, movies, cosine_sim):

    indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).drop_duplicates()

    idx = indices[title]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )[1:11]

    movie_indices = [i[0] for i in similarity_scores]

    return movies.iloc[movie_indices]