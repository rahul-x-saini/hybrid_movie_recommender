import streamlit as st
import requests
import pandas as pd


movies = pd.read_csv("data/movies.csv")
movies = movies.sort_values("title")

API_URL = "http://127.0.0.1:8000/recommend"

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Movie Recommender 🎬",
    layout="wide"
)

# -----------------Title--------------------

st.title("🎬 Movie Recommendation System")
st.markdown("Hybrid + Ranking based Movie Recommender")
st.write("---")

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ User Inputs")

# Project Info / Instructions
st.sidebar.markdown(
    """
    Welcome!   
    This recommender suggests movies based on your favorite choice.  

    **Steps:**  
    1. Enter your **User ID**  
    2. Select a **movie you like**  
    3. Click **Get Recommendations**  
    """
)

st.sidebar.write("---")  


user_id = st.sidebar.number_input("User ID", min_value=1, value=1)


favorite_movie = st.sidebar.selectbox("Select a Movie", 
            movies["title"].tolist()
            )

get_recommendations = st.sidebar.button("Get Recommendations")

st.sidebar.markdown("---")
st.sidebar.caption("ML Project Demo | Clean & Professional UI")

# ---------------- MAIN PANEL ----------------

if get_recommendations:
    with st.spinner("Fetching recommendations..."):
        params = {"user_id": user_id, "favorite_movie": favorite_movie}
        response = requests.get(API_URL, params=params)

        if response.status_code == 200:
            data = response.json()

            if len(data) == 0:
                st.warning("No recommendations found for this movie.")
            else:
                df = pd.DataFrame(data)
                st.subheader("🎯 Recommended Movies")
                for i, row in df.iterrows():
                    st.markdown(f"🎬 **{row['title']}** | {row['genres']}")
        else:
            st.error("❌ Failed to fetch recommendations from API")