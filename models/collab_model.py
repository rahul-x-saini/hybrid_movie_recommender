import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(self, ratings_path, factors=20, lr=0.01, reg=0.02, epochs=50):
        self.ratings = pd.read_csv(ratings_path)
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.user_map = {}
        self.item_map = {}
        self.P = None
        self.Q = None

    def prepare_data(self):
        users = self.ratings['userId'].unique()
        items = self.ratings['movieId'].unique()


        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}

        self.num_users = len(users)
        self.num_items = len(items)

        self.P = np.random.normal(scale=1./self.factors, size=(self.num_users, self.factors))
        self.Q = np.random.normal(scale=1./self.factors, size=(self.num_items, self.factors))

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0

            for _, row in self.ratings.iterrows():
                u = self.user_map[row['userId']]
                i = self.item_map[row['movieId']]
                r = row['rating']

                p_u = self.P[u].copy()
                q_i = self.Q[i].copy()

                prediction = np.dot(p_u, q_i)
                error = r - prediction
                total_loss += error**2

                
                self.P[u] += self.lr * (error * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (error * p_u - self.reg * q_i)

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.2f}")

    def recommend(self, user_id, top_n=100):
        if user_id not in self.user_map:
            return []

        u = self.user_map[user_id]

        scores = np.dot(self.P[u], self.Q.T)
        top_indices = np.argsort(scores)[-top_n:][::-1]

        reverse_item_map = {v: k for k, v in self.item_map.items()}
        return [(int(reverse_item_map[i]), float(scores[i])) for i in top_indices]
