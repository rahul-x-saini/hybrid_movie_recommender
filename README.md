# Hybrid Recommendation System 
## (Content-Based + collaborative-Based + Pairwise Ranking)

A hybrid recommendation system built from scratch using NumPy, combining collaborative and content-based filtering with pairwise ranking.

Mitigates cold-start via popular item recommendations and evaluates performance using Precision@K, Recall@K, NDCG@K, and Hit Rate@K.

FastAPI backend for inference APIs
Streamlit UI for interactive recommendations

                ┌──────────────────────┐
                │     Streamlit UI     │
                │   (User Interface)   │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │       FastAPI        │
                │   REST API Layer     │
                └─────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌──────────────────────┐     ┌────────────────────────┐
│   Content-Based      │     │   Collaborative-Based  │
│   (Item Features)    │     │   (User-Item Matrix)   │
└──────────┬───────────┘     └──────────┬─────────────┘
           └──────────────┬─────────────┘
                          ▼
                ┌──────────────────────┐
                │    Hybrid Engine     │
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ Pairwise Ranking     │
                │ (Learning Model)     │
                └─────────┬────────────┘
                          ▼
                Ranked Recommendations



# RUN THESE COMMANDS TO RUN THIS PROJECT ON YOUR OWN MACHINE


git clone https://github.com/rahul-x-saini/hybrid-recommender.git
cd hybrid_movie_recommender
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m uvicorn api.server:app --reload
streamlit run streamlit/app.py
