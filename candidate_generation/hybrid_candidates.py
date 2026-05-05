import pandas as pd
import numpy as np
from models.content_model import ContentRecommender
from models.collab_model import MatrixFactorization


class HybridCandidateGenerator:
    def __init__(self, movies_path="data/movies.csv", ratings_path="data/train.csv", mode="train", alpha=0.5):
        
        self.movies = pd.read_csv(movies_path)
        self.alpha = alpha 

        if mode == "train":
            self.content_model = self._build_content_model(self.movies)
            self.collab_model = self._build_collab_model(ratings_path)
        else:
            self.content_model = self._load_content_model()
            self.collab_model = self._load_collab_model()

    # ---------- Build ----------
    def _build_content_model(self, movies):
        model = ContentRecommender(movies)
        model.prepare_matrix()
        return model

    def _build_collab_model(self, ratings_path):
        model = MatrixFactorization(ratings_path, epochs=50)
        model.prepare_data()
        model.train()
        return model

    # ---------- Load ----------
    def _load_content_model(self):
        import joblib
        return joblib.load("models/content_model.pkl")

    def _load_collab_model(self):
        import joblib
        return joblib.load("models/collab_model.pkl")

    # ---------------Content Based Recommendations-----------------

    def content_recs(self, favorite_title, top_n=100):
        movie_indices = self.content_model.recommend(favorite_title, top_n)
        if not movie_indices:
            return pd.DataFrame(columns=['movieId', 'content_score'])

        movie_ids = self.movies.iloc[movie_indices]['movieId'].values

        recs = pd.DataFrame({
           'movieId': movie_ids,
           'content_score': np.arange(len(movie_ids), 0, -1) / len(movie_ids)
           })
        return recs

    # ---------------Collaborative Recommendations-----------------

    def collab_recs(self, user_id, top_n=100):

        recs = self.collab_model.recommend(user_id, top_n=top_n)

        if not recs:
            return pd.DataFrame(columns=['movieId', 'collab_score'])
        recs_df = pd.DataFrame(recs, columns=['movieId', 'collab_score'])
        recs_df['collab_score'] = recs_df['collab_score'].rank(pct=True)
        return recs_df

    # -------------Hybrid Recommendations--------------

    def generate_candidates(self, user_id, favorite_movie_title, top_n=100):
        content = self.content_recs(favorite_movie_title, top_n)
        collab = self.collab_recs(user_id, top_n)
        
        if content.empty and collab.empty:
            return pd.DataFrame(columns=['movieId', 'content_score','collab_score',"hybrid_score" ])
       
        candidates = pd.merge(content, collab, on='movieId', how='outer', validate='one_to_one')
        candidates = candidates.fillna({
           'content_score': 0,
           'collab_score': 0
           })

        candidates = candidates.merge(self.movies, on='movieId', how='left')
        candidates['hybrid_score'] = self.alpha * candidates['content_score'] + (1 - self.alpha) * candidates['collab_score']
        candidates = candidates.sort_values('hybrid_score', ascending=False)
        return candidates

    # ----------Cold Start Handling------------


    def handle_cold_start(self, top_n=100):

        movie_stats = self.collab_model.ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
        movie_stats['score'] = np.log1p(movie_stats['count']) * movie_stats['mean']
        top_movies = movie_stats.sort_values('score', ascending=False).head(top_n).index

        movies = self.movies[self.movies['movieId'].isin(top_movies)].copy()
        movies = movies.set_index('movieId').loc[top_movies].reset_index()
        return movies
