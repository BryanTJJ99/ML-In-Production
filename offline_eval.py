import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib

# Load data
df = pd.read_csv('ratings.csv', names=['date', 'userId', 'movieId', 'rating'])
df = df.iloc[:, [1, 2, 3]]
k = 20  # Number of neighbors
n = 20  # Number of recommendations
top_rated_movies = df[df['rating'] == 5]['movieId'].unique()

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
        print('New User. Here are some movies rated 5/5!')
        return np.random.choice(top_rated_movies, n).tolist()

    sp_matrix_user_index = list(train_matrix.index).index(user)
    test_data = csr_matrix(train_matrix.values[sp_matrix_user_index].reshape(1, -1))
    print('Generating recommendations...')
    distances, indices = model.kneighbors(test_data, n_neighbors=k)
    top_k_users_indices = indices.flatten()
    similar_users = train_matrix.index[top_k_users_indices].tolist()
    similar_user_ratings = df[df['userId'].isin(similar_users)]
    recommended_movies = similar_user_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).index[:n]
    print(f"Top {n} movie recommendations for user:")
    return list(recommended_movies)


def movie_recs(model, user, trainBool=False, train_matrix=None, val_matrix=None):
    model = train(trainBool, model, train_matrix)
    return test(user, model, k, train_matrix)


def precision_at_k(user, model, K, n, train_matrix, val_matrix, top_rated_movies):
    relevant_count = 0
    correct_count = 0
    predictions = movie_recs(model, user, False, train_matrix, val_matrix)
    
    for prediction in predictions[:K]:
        if (prediction in val_matrix.loc[user].index and val_matrix.loc[user, prediction] != 0) or prediction in top_rated_movies:
            correct_count += 1

    precision = correct_count / min(K, n)
    return precision

# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
train_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
val_matrix = val_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# MODEL = train(True, MODEL, train_matrix)

# Initialize a list to store the precision and accuracy values
precisions = []
accuracies = []

# Calculate precision and accuracy at k for validation users
K = 20  # Top K recommendations
val_users = val_matrix.index
for user in val_users:
    precision = precision_at_k(user, MODEL, K, n, train_matrix, val_matrix, top_rated_movies)
    precisions.append(precision)

# Calculate the average precision and accuracy at k
average_precision_at_k = np.mean(precisions)

# Print the average precision and accuracy at k
print(f"For {len(val_users)} users in the validation set, the average precision at {K} is {average_precision_at_k}.")