from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

import joblib

def train_svd_model(ratings):

    reader = Reader(rating_scale=(0.5,5))

    data = Dataset.load_from_df(
        ratings[["userId","movieId","rating"]],
        reader
    )

    trainset,testset = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    model = SVD()

    model.fit(trainset)

    joblib.dump(model,"models/svd_model.pkl")

    return model


def recommend_for_user(user_id,
                       model,
                       ratings,
                       movies):

    watched = ratings[
        ratings.userId==user_id
    ]["movieId"]

    unseen = movies[
        ~movies.movieId.isin(watched)
    ]

    predictions=[]

    for movie in unseen.movieId:

        pred=model.predict(user_id,movie)

        predictions.append(
            (movie,pred.est)
        )

    predictions.sort(
        key=lambda x:x[1],
        reverse=True
    )

    ids=[i[0] for i in predictions[:10]]

    return movies[
        movies.movieId.isin(ids)
    ]