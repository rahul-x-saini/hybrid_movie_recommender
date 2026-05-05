import pandas as pd


def remove_seen_movies(recommendations, user_id, ratings):

    seen = ratings[ratings["userId"] == user_id]["movieId"]

    recommendations = recommendations[
        ~recommendations["movieId"].isin(seen)
    ]

    return recommendations


def diversify_genres(recommendations, max_per_genre=3):

    selected = []
    genre_count = {}

    for _, row in recommendations.iterrows():
       
        if pd.notna(row["genres"]):
            genres = row["genres"].split("|")
            main_genre = genres[0]

        else:
            continue       

        
        if genre_count.get(main_genre, 0) < max_per_genre:
            selected.append(row)
            genre_count[main_genre] = genre_count.get(main_genre, 0) + 1

    return pd.DataFrame(selected)