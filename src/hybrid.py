import pandas as pd

from src.content_based import recommend_movies
from src.collaborative import recommend_for_user


def hybrid_recommend(
    user_id,
    movie_name,
    movies,
    ratings,
    cosine_sim,
    model
):

    content = recommend_movies(
        movie_name,
        movies,
        cosine_sim
    )

    collaborative = recommend_for_user(
        user_id,
        model,
        ratings,
        movies
    )

    final = pd.concat(
        [content,collaborative]
    )

    final = final.drop_duplicates(
        subset="title"
    )

    return final.head(10)