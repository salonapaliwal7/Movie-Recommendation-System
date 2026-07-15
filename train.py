from src.utils import *

from src.content_based import *

from src.collaborative import *

movies,ratings,tags,links = load_data()

movies = preprocess_movies(movies)

cosine = train_content_model(movies)

svd = train_svd_model(ratings)

print("Training Complete")