import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.cosine_sim = None

    def prepare_matrix(self):

        self.movies['content'] = self.movies['title'] + " " + self.movies['genres']

        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, favorite_title, top_n=50):
        if self.cosine_sim is None:
            raise Exception("Run prepare_matrix() first")

        idx = self.movies[self.movies['title'] == favorite_title].index

        if len(idx) == 0:
            return []

        idx = idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        movie_indices = [i[0] for i in sim_scores]

        return movie_indices