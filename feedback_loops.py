import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib
import ast
import tqdm

# Load data
df = pd.read_csv('ratings.csv', names=['time', 'userId', 'movieId', "id", 'rating'])

movies_data = pd.read_csv("movies.csv", names=['id','tmdb_id', "vote_avg", 'vote_count','title','language','adult','genres','runtime','popularity', 'temp'])

k = 20  # Number of neighbors
n = 20  # Number of recommendations
top_rated_movies = df[df['rating'] == 5]['movieId'].unique()

# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
train_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
val_matrix = val_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Model loading
model_path = 'nearest_neigh_offline_eval.joblib'
try:
    MODEL = joblib.load(model_path)
except FileNotFoundError:
    MODEL = None

def train(bool, model, train_matrix):
    if bool:
        print('Training a new model..')
        train_sparse_matrix = csr_matrix(train_matrix.values)
        model = NearestNeighbors(metric='cosine', algorithm="brute", n_jobs=-1)
        model.fit(train_sparse_matrix)
        joblib.dump(model, model_path)
        print('Model training & saving complete!')
    return model

def test(user, model, k, train_matrix):
    if user not in train_matrix.index:
        # print('New User. Here are some movies rated 5/5!')
        return np.random.choice(top_rated_movies, n).tolist()

    sp_matrix_user_index = list(train_matrix.index).index(user)
    test_data = csr_matrix(train_matrix.values[sp_matrix_user_index].reshape(1, -1))
    # print('Generating recommendations...')
    distances, indices = model.kneighbors(test_data, n_neighbors=k)
    top_k_users_indices = indices.flatten()
    similar_users = train_matrix.index[top_k_users_indices].tolist()
    similar_user_ratings = df[df['userId'].isin(similar_users)]
    recommended_movies = similar_user_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).index[:n]
    # print(f"Top {n} movie recommendations for user:")
    return list(recommended_movies)


def movie_recs(model, user, trainBool=False, train_matrix=None, val_matrix=None):
    model = train(trainBool, model, train_matrix)
    # print(test(user, model, k, train_matrix))
    return test(user, model, k, train_matrix)


def calculate_diversity_score(recommendations, ratings_data, movies_data):
    genre_counts = {}
    total_recommendations = len(recommendations)
    
    # Iterate over each recommended movie
    for movie_id in recommendations:
        # Fetch the corresponding movie ID from ratings data
        movie_id_in_ratings = ratings_data[ratings_data['movieId'] == movie_id]['id'].iloc[0]
        # print(movie_id_in_ratings)
        # Fetch genres for the movie from movies data using the corresponding movie ID
        # print(movies_data[movies_data['id'] == movie_id_in_ratings])
        movie_genres = movies_data[movies_data['id'] == movie_id_in_ratings]['genres'].iloc[0]
        genres_list = ast.literal_eval(movie_genres)  # Convert string representation of list of dictionaries to actual list of dictionaries
        # Increment genre counts for each genre in the movie
        for genre in genres_list:
            genre_name = genre
            genre_counts[genre_name] = genre_counts.get(genre_name, 0) + 1
    
    # Calculate diversity score as a weighted average
    if total_recommendations > 0:
        diversity_score = sum(1 / count for count in genre_counts.values()) / total_recommendations
    else:
        diversity_score = 0.0
    
    return diversity_score

# Example usage
# recommendations = movie_recs(MODEL, 799, trainBool=False, train_matrix=train_matrix, val_matrix=val_matrix)

# diversity_score = calculate_diversity_score(recommendations, df, movies_data)
# print("Diversity score:", diversity_score)

def calculate_diversity_scores_for_all_users(val_df, model, train_matrix, movies_data):
    all_diversity_scores = {}
    val_users = val_df['userId'].unique()
    for user_id in tqdm.tqdm(val_users):
        recommendations = movie_recs(model, user_id, trainBool=False, train_matrix=train_matrix, val_matrix=val_matrix)
        diversity_score = calculate_diversity_score(recommendations, df, movies_data)
        all_diversity_scores[user_id] = diversity_score
    return all_diversity_scores

# Example usage
all_diversity_scores = calculate_diversity_scores_for_all_users(val_df, MODEL, train_matrix, movies_data)
# Calculate average diversity score
average_diversity_score = np.mean(list(all_diversity_scores.values()))
print("Average diversity score:", average_diversity_score)



