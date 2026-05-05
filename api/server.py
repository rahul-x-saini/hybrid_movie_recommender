from fastapi import FastAPI
from pipeline.recommend import recommend, load_models

app = FastAPI()

ranker, hybrid_gen, ratings = load_models()

@app.get("/")
def home():
    return {"Message": "Movie Recommender is running"}


@app.get("/recommend")
def get_recommendations(user_id: int, favorite_movie: str):
    rec = recommend(
        user_id,
        favorite_movie,
        ranker,
        hybrid_gen,
        ratings
    )
    return rec.to_dict(orient="records")
