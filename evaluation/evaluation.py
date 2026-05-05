import numpy as np
import pandas as pd
from ranking.features import build_features

def precision_at_k(recommended, relevant, k=10):
    """Precision@K"""
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Recall@K"""
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0


def dcg_at_k(recommended, relevance, k=10):
    recommended = recommended[:k]

    return sum(
        (2**relevance.get(item, 0) - 1) / np.log2(i + 2)
        for i, item in enumerate(recommended)
    )

def idcg_at_k(relevance, k=10):
    sorted_rels = sorted(relevance.values(), reverse=True)
    ideal = sorted_rels[:k]

    return sum(
        (2**rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ideal)
    )

def ndcg_at_k(recommended, relevance, k=10):
    ideal = idcg_at_k(relevance, k)

    if ideal == 0:
        return 0

    actual = dcg_at_k(recommended, relevance, k)

    return actual / ideal

def hit_rate_at_k(recommended, relevant, k):
    if not relevant:
        return 0
    return int(len(set(recommended[:k]) & set(relevant)) > 0)


def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0



def evaluate_recommender(test_df, users, hybrid_gen, ranking_model=None, top_k=10):
    """
    Evaluate baseline (popularity), hybrid, and ranking models.
    
    Parameters:
        test_df: pd.DataFrame -> test set with columns ['userId','movieId','rating']
        users: list -> user IDs to evaluate
        hybrid_gen: HybridCandidateGenerator object
        ranking_model: RankingModel object (optional)
        top_k: int -> number of top recommendations
    """
    #--------Popularity baseline---------

    popularity = hybrid_gen.collab_model.ratings.groupby("movieId")['rating'].count().sort_values(ascending=False)
    popular = popularity.index.tolist()
    ratings = hybrid_gen.collab_model.ratings
    user_groups = test_df.groupby('userId')
    train_groups = ratings.groupby('userId')
    user_seen = train_groups['movieId'].apply(set)

    precision_baseline, precision_hybrid, precision_rank = [], [], []
    recall_baseline, recall_hybrid, recall_rank = [], [], []
    ndcg_baseline, ndcg_hybrid, ndcg_rank = [], [], []
    hit_baseline, hit_hybrid, hit_rank = [], [], []

    non_cold_users = 0
    rank_count = 0
    evaluated_users = 0
    cold_users = 0

    for uid in users:
        
        user_data = user_groups.get_group(uid)
        user_data = user_data[user_data['rating'] >= 4.0]

        if len(user_data) == 0:
            continue

        relevant = user_data['movieId'].tolist()
        relevant_ = dict(zip(user_data['movieId'], user_data['rating']))
        evaluated_users += 1
        seen = user_seen.get(uid, set())

        #-----------Baseline-----------

        recommended_baseline = [m for m in popular if m not in seen][:top_k]

        
        precision_baseline.append(precision_at_k(recommended_baseline, relevant, top_k))
        recall_baseline.append(recall_at_k(recommended_baseline, relevant, top_k))
        ndcg_baseline.append(ndcg_at_k(recommended_baseline, relevant_, top_k))
        hit_baseline.append(hit_rate_at_k(recommended_baseline, relevant, top_k))

        #--------------Hybrid candidate-------------

        cold_start = None
        if uid in train_groups.groups:

            train_user = train_groups.get_group(uid)
            fav_movie_id = train_user.sort_values('rating', ascending=False).iloc[0]['movieId']
            cold_start = False
        else:
            fav_movie_id = None 
            cold_start = True


        if fav_movie_id is not None:

            fav_movie = hybrid_gen.movies[
                hybrid_gen.movies['movieId'] == fav_movie_id
            ]['title'].values
            fav_movie = fav_movie[0] if len(fav_movie) > 0 else None

        else:
            fav_movie = None

        candidates = None

        if cold_start:
            cold_users += 1
            candidates = hybrid_gen.handle_cold_start(top_n=100)
            recommended_hybrid = candidates['movieId'].tolist()

        else:
            candidates = hybrid_gen.generate_candidates(uid, fav_movie, top_n=100)

            recommended_hybrid = candidates['movieId'].tolist()
            non_cold_users += 1


        recommended_hybrid = [
            m for m in recommended_hybrid if m not in seen
        ][:top_k]

        precision_hybrid.append(precision_at_k(recommended_hybrid, relevant, top_k))
        recall_hybrid.append(recall_at_k(recommended_hybrid, relevant, top_k))
        ndcg_hybrid.append(ndcg_at_k(recommended_hybrid, relevant_, top_k))
        hit_hybrid.append(hit_rate_at_k(recommended_hybrid, relevant, top_k))


        #----------------Ranking-----------------

        if ranking_model is not None:

            if cold_start:
                continue

            rank_count += 1

            #----------feature engineering-----------

            X = build_features(candidates)

            scores = ranking_model.rank(X)

            candidates = candidates.copy()
            candidates["rank_score"] = scores

            candidates = candidates.sort_values("rank_score", ascending=False)

            recommended_rank = candidates['movieId'].tolist()

            recommended_rank = [
                  m for m in recommended_rank if m not in seen
            ][:top_k]
            
            precision_rank.append(precision_at_k(recommended_rank, relevant, top_k))
            recall_rank.append(recall_at_k(recommended_rank, relevant, top_k))
            ndcg_rank.append(ndcg_at_k(recommended_rank, relevant_, top_k))
            hit_rank.append(hit_rate_at_k(recommended_rank, relevant, top_k))

    results = {
        "k": top_k,
        "Ranking_users": rank_count,
        "Hybrid_users": non_cold_users,
        "cold_users":cold_users,
        "Evaluated_users":evaluated_users,

        "Precision@K": {
            "Baseline": safe_mean(precision_baseline),
            "Hybrid": safe_mean(precision_hybrid),
            "Ranking": safe_mean(precision_rank) if ranking_model else None
        },
        "Recall@K": {
            "Baseline": safe_mean(recall_baseline),
            "Hybrid": safe_mean(recall_hybrid),
            "Ranking": safe_mean(recall_rank) if ranking_model else None
        },
        "NDCG@K": {
            "Baseline": safe_mean(ndcg_baseline),
            "Hybrid": safe_mean(ndcg_hybrid),
            "Ranking": safe_mean(ndcg_rank) if ranking_model else None
        },
        "HitRate@K": {
            "Baseline": safe_mean(hit_baseline),
            "Hybrid": safe_mean(hit_hybrid),
            "Ranking": safe_mean(hit_rank) if ranking_model else None
        }
    }
    return results
