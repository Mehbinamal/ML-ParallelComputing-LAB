import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#part B
anime = pd.read_csv('Anime_Dataset/anime.csv')
rating = pd.read_csv('Anime_Dataset/cleaned_rating.csv')

print(anime.head())
print(rating.head())

data = pd.merge(rating, anime, on="anime_id")
print(data.head())

user_item_matrix = data.pivot_table(
    index='user_id',
    columns='name',
    values='rating_x'
)

print(user_item_matrix.head())

user_item_matrix = user_item_matrix.fillna(0)

#part C
user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print(user_similarity_df.head())

def recommend_user_based(user_id):

    similar_users = user_similarity_df[user_id]

    weighted_sum = user_item_matrix.T.dot(similar_users)

    similarity_sum = similar_users.sum()

    predicted_ratings = weighted_sum / similarity_sum

    recommended = predicted_ratings.sort_values(ascending=False)

    return recommended.head(5)

print(recommend_user_based(3))

