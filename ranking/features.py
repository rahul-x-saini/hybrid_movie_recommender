import pandas as pd
import numpy as np

ratings = pd.read_csv("data/train.csv")
movie_pop = ratings.groupby('movieId')['rating'].count()
GLOBAL_POP_MAX = np.log1p(movie_pop).max()


def safe_norm(df, col):
        max_val = df[col].max()
        return df[col] / max_val if max_val > 0 else df[col]

def build_features(candidates):
    """
    Build ranking features using global dataset statistics for inference
    """
    df = candidates.copy()
    # -------------------------
    # Movie popularity
    # -------------------------

    df['popularity'] = df['movieId'].map(movie_pop).fillna(0)
    df['popularity'] = np.log1p(df['popularity'])

    # -------------------------
    # score normalization
    # -------------------------

    df['hybrid_score'] = safe_norm(df, 'hybrid_score')
    df['content_score'] = safe_norm(df, 'content_score')
    df['collab_score'] = safe_norm(df, 'collab_score')
    df['collab_x_content'] = df['collab_score'] * df['content_score']

    df['popularity'] = df['popularity'] / GLOBAL_POP_MAX

    return df[['hybrid_score','content_score','collab_score','popularity','collab_x_content']]


def prepare_features(candidates, user_history):
    """
    Build training features + labels for training ranking model 
    """
    df = candidates.copy()

    # -------------------------
    # popularity feature
    # -------------------------

    df['popularity'] = df['movieId'].map(movie_pop).fillna(0)
    df['popularity'] = np.log1p(df['popularity'])


    # -------------------------
    # safe normalization 
    # -------------------------


    df['hybrid_score'] = safe_norm(df, 'hybrid_score')
    df['content_score'] = safe_norm(df, 'content_score')
    df['collab_score'] = safe_norm(df, 'collab_score')

    df['collab_x_content'] = df['collab_score'] * df['content_score']
    df['popularity'] = df['popularity'] / GLOBAL_POP_MAX
    
    # -------------------------
    # label creation (vectorized)
    # -------------------------
    rating_map = user_history.set_index('movieId')['rating']

    df['label'] = df['movieId'].map(rating_map)

    df['label'] = (df['label'] >= 4).fillna(0).astype(int)

    features = df[['hybrid_score','content_score','collab_score','popularity','collab_x_content']]
    labels = df['label']

    return features, labels

