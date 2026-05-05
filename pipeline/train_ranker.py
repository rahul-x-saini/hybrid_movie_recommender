import pandas as pd
import mlflow
import mlflow.xgboost

from candidate_generation.hybrid_candidates import HybridCandidateGenerator
from ranking.features import prepare_features
from ranking.ranker import RankingModel
from pipeline.save_models import save_models
from pipeline.save_results import save_results_csv
from evaluation.evaluation import evaluate_recommender

# ---------------------------------------------------
# ----------Traning and Saving Ranker Model----------
# ---------------------------------------------------


def train_ranker():
    
    mlflow.set_tracking_uri("file:///C:/mlruns")
    mlflow.set_experiment("movie_recommender_system")

    with mlflow.start_run():


        # ---------------- MLflow Params ----------------
        mlflow.log_param("model_type", "XGBRanker")
        mlflow.log_param("alpha_hybrid", 0.5)
        mlflow.log_param("top_k", 10)

        # Initialize candidate generator
        print("Initializing hybrid candidate generator...")

        movies_path = "data/movies.csv"
        ratings_path = "data/train.csv"

        hybrid_gen = HybridCandidateGenerator(
            movies_path,
            ratings_path,
            alpha=0.5
        )

        ratings = hybrid_gen.collab_model.ratings
        user_groups = ratings.groupby('userId')

        # Ranking model
        ranker = RankingModel()

        # ---------------- MLflow Model Params ----------------
        params = {
            "objective": "rank:pairwise",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

        mlflow.log_params(params)

        X_train = []
        y_train = []
        groups = []

        users = ratings['userId'].unique()

        print("Generating training data for ranking model...")

        for user_id in users:

            user_history = user_groups.get_group(user_id)

            fav_movie_id = user_history.sort_values('rating', ascending=False).iloc[0]['movieId']
        
            fav_movie = hybrid_gen.movies[
                hybrid_gen.movies['movieId'] == fav_movie_id
                ]['title'].values

            fav_movie = fav_movie[0] if len(fav_movie) > 0 else None

            # Candidate generation
            candidates = hybrid_gen.generate_candidates(
                user_id,
                fav_movie,   
                top_n=100
            )
        
            if candidates.empty:
                continue
            # Build features + labels
            X, y = prepare_features(candidates, user_history)

            X_train.append(X)
            y_train.append(y)

            groups.append(len(X))


        X_train = pd.concat(X_train)
        y_train = pd.concat(y_train)

        print("Training ranking model...")

        ranker.train(X_train, y_train, groups)

        print("Ranking model training completed.")


        # ---------------- MLflow Metrics ----------------

        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("num_users", len(users))

        # ---------------- MLflow Model Logging ----------------
        mlflow.xgboost.log_model(
            ranker.model,
            name="ranking_model"
        )


        # ---------------- Evaluation ----------------

        test_df = pd.read_csv("data/test.csv")
        test_users = test_df['userId'].unique()
  
        print("Evaluating .....")

        results = evaluate_recommender(
            test_df,
            test_users, 
            hybrid_gen, 
            ranker, 
            top_k=10
        )

        print("Evaluation Process Completed")

        # ---------------- MLflow Evaluation Metrics ----------------

        mlflow.log_metric("ranking_users", results["Ranking_users"])
        mlflow.log_metric("hybrid_users", results["Hybrid_users"])
        mlflow.log_metric("cold_users", results["cold_users"])
        mlflow.log_metric("evaluated_users", results["Evaluated_users"])

        mlflow.log_metric("precision_at_k_baseline", results["Precision@K"]["Baseline"])
        mlflow.log_metric("precision_at_k_hybrid", results["Precision@K"]["Hybrid"])
        mlflow.log_metric("precision_at_k_ranking", results["Precision@K"]["Ranking"])
        
        mlflow.log_metric("recall_at_k_baseline", results["Recall@K"]["Baseline"])
        mlflow.log_metric("recall_at_k_hybrid", results["Recall@K"]["Hybrid"])
        mlflow.log_metric("recall_at_k_ranking", results["Recall@K"]["Ranking"])

        mlflow.log_metric("ndcg_at_k_baseline", results["NDCG@K"]["Baseline"])
        mlflow.log_metric("ndcg_at_k_hybrid", results["NDCG@K"]["Hybrid"])
        mlflow.log_metric("ndcg_at_k_ranking", results["NDCG@K"]["Ranking"])

        mlflow.log_metric("hit_rate_at_k_baseline", results["HitRate@K"]["Baseline"])
        mlflow.log_metric("hit_rate_at_k_hybrid", results["HitRate@K"]["Hybrid"])
        mlflow.log_metric("hit_rate_at_k_ranking", results["HitRate@K"]["Ranking"])
        

        # ---------------- Save + Log Results ----------------

        file_path = save_results_csv(results)
        mlflow.log_artifact(file_path)

        save_models(ranker, hybrid_gen)
