from pipeline.postprocess import remove_seen_movies, diversify_genres
from candidate_generation.hybrid_candidates import HybridCandidateGenerator
from ranking.features import build_features
import joblib
import pandas as pd

# ----------------------------------
# ----------Loading Models----------
# ----------------------------------

def load_models():

    ranker = joblib.load("models/ranking_model.pkl")

    hybrid_gen = HybridCandidateGenerator(
        mode="inference"
    )

    ratings = hybrid_gen.collab_model.ratings

    return ranker, hybrid_gen, ratings


# ---------------------------------------------
# ----------Recommendation Generation----------
# ---------------------------------------------


def recommend(user_id, favorite_movie, ranker, hybrid_gen, ratings):
    
    user_exists = True
    #-------------candidate generation----------------
    if user_id not in hybrid_gen.collab_model.user_map:
        print("cold_user")
        user_exists = False
        candidates = hybrid_gen.handle_cold_start( 
            top_n=100
        )

    else:
        print("non_cold_user")
        candidates = hybrid_gen.generate_candidates(
            user_id,
            favorite_movie,
            top_n=100
        )
    
    if candidates.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    # ------------Pre-filter-------------

    if user_exists:
        candidates = remove_seen_movies(candidates, user_id, ratings)

        #---------feature engineering----------

        X = build_features(candidates)

        #-------------ranking-------------

        candidates["rank_score"] = ranker.rank(X)

        candidates = candidates.sort_values(
           "rank_score",
            ascending=False
        )

        #------------POST PROCESSING----------

    candidates = diversify_genres(candidates)

    
    return candidates[["movieId", "title", "genres"]].head(10)

