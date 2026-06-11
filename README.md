
# Hybrid Movie Recommender System

An end-to-end **hybrid recommendation system** combining collaborative filtering, content-based filtering, and learning-to-rank for personalized movie recommendations.

---

## Features

- Hybrid recommendation system (Collaborative + Content-based)
- Matrix Factorization (from scratch)
- TF-IDF based content similarity
- Learning-to-Rank model (XGBoost Ranker)
- Cold-start handling for new users
- Evaluation using ranking metrics (Precision@K, Recall@K, NDCG, Hit Rate)
- MLflow tracking
- FastAPI backend + Streamlit UI
- Dockerized for reproducible deployment
- Deployed on Google Cloud Platform (GCP)

---


## Models Used

- Collaborative Filtering: Matrix Factorization
- Content-Based Filtering: TF-IDF + Cosine Similarity
- Ranking Model: XGBoost Ranker

---

## Evaluation Metrics

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate@K

Compared across:
- Baseline (Popularity)
- Hybrid Model
- Ranking Model

---

## Mlflow Tracking

- MLflow is used for experiment tracking (`file:///C:/mlruns`) under the experiment `movie_recommender_system`, logging model hyperparameters, training statistics, evaluation metrics (Precision@K, Recall@K, NDCG@K, Hit Rate@K), user segmentation results, and the trained XGBoost ranking model along with evaluation artifacts.

## ⚙️ Tech Stack

- Python, NumPy, Pandas
- Scikit-learn, XGBoost
- FastAPI
- Streamlit
- Mlflow
- Docker
- Google Cloud Platform (GCP)

---

## Deployment

- Deployed as a fully containerized application using Docker, with the backend hosted on a GCP VM and the frontend deployed on Cloud Run.

---

### System Architecture

``` text
                ┌──────────────────────┐                     
                │      Streamlit UI    │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │       FastAPI        │     
                └─────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌──────────────────────┐     ┌────────────────────────┐
│   Content-Based      │     │   Collaborative-Based  │
│ (TF-IDF Similarity)  │     │ (Matrix Factorization) │
└──────────┬───────────┘     └──────────┬─────────────┘
           └──────────────┬─────────────┘
                          ▼
                ┌──────────────────────────┐
                │ Hybrid Candidate Engine  │
                │ (Merge + Scoring)        │
                └─────────┬────────────────┘
                          ▼
                ┌──────────────────────────┐
                │ Learning-to-Rank Model   │
                │ (XGBoost Ranker)         │
                └─────────┬────────────────┘
                          ▼
                Ranked Recommendations
```


# RUN THESE COMMANDS TO RUN THIS PROJECT ON YOUR OWN MACHINE


git clone https://github.com/rahul-x-saini/hybrid_movie_recommender.git

cd hybrid_movie_recommender

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python -m uvicorn api.server:app --reload

streamlit run streamlit/app.py
