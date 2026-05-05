import pandas as pd

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

print(ratings.info())
print(ratings.isna().sum())

print(movies.info())

print("\nUnique Users:", ratings['userId'].nunique())
print("Unique Movies:", ratings['movieId'].nunique())

print("\nRating Distribution:")
print(ratings['rating'].value_counts().sort_index())

num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()


num_ratings = len(ratings)

sparsity = 1 - (num_ratings / (num_users * num_movies))
print("\nSparsity:", round(sparsity * 100, 2), "%")

# Filter users with at least 50 ratings
user_counts = ratings.groupby('userId').size()
print(user_counts)

active_users = user_counts[user_counts >= 50].index
print(active_users)

filtered_ratings = ratings[ratings['userId'].isin(active_users)]
print(filtered_ratings)


# Now sample 5000 users for manageable dataset
print(filtered_ratings['userId'].drop_duplicates())

sample_users = filtered_ratings['userId'].drop_duplicates().sample(5000, random_state=42)
print(sample_users)

subset = filtered_ratings[filtered_ratings['userId'].isin(sample_users)]
print("Final subset shape:", subset.shape)

# Save subset
subset.to_csv("data/ratings_subset.csv", index=False)

print("Unique users:", subset['userId'].nunique())
print("Unique movies:", subset['movieId'].nunique())
