import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.model_selection import cross_val_score
import pickle
import argparse
from scipy.sparse import csr_matrix
import joblib
import time

# Logic Attribution: Code inspiration and logic taken from Copilot AI companion, in accordance with course policy
df = pd.read_csv('ratings.csv', names = ['date', 'userId', 'movieId', 'rating'])
df = df.iloc[:,[1,2,3]]
df_movies = pd.read_csv('movies.csv', names = ['movieId', 'num1', 'num2', 'title', 'language', 'bool?', 'genre', 'num3', 'num4'])
df_movies = df_movies.iloc[:,[0,6]]
k = 20
# no of recommendations
n = 20
# for new users recommend n movies randomly which are rated 5
top_rated_movies = df[df['rating'] == 5]['movieId'].unique()

# model loading for CLI testing - statement not used for deployed version
# loaded in flask for faster prediction 
model = joblib.load('nearest_neigh_model.joblib')

# user-movie matrix
# if user hasn't rated a movie value is zero
matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
# convert it to sparse format
sparse_matrix = csr_matrix(matrix.values)

def train(bool, model):
    if bool:
        start = time.time()
        print('Training a new model..')
        # cosine similarity
        # nearest neighbor model, n_jobs for parallelization
        model =  NearestNeighbors(metric='cosine',  algorithm = "brute", n_jobs=-1)
        model.fit(sparse_matrix)
        # save model to the file, replaces existing model!
        end = time.time()
        joblib.dump(model, 'nearest_neigh_model.joblib')
        print('Model training & saving complete!')
        print('Time taken to train:', end-start)
    
    return model


def predict(user, model, k):
    # check if user is present in the data a.k.a old user
    # if new user, we have no prior data so we recommend n top rated movies randomly
    start = time.time()
    if user not in matrix.index:
        print('New User. Here are some movies rated 5/5!')
        return np.random.choice(top_rated_movies, n).tolist()
    
    # convert the pivot table indexing (same as user) to csr matrix indexing
    sp_matrix_user_index = list(matrix.index).index(user)
    # data point to predict
    test_data = sparse_matrix[sp_matrix_user_index].reshape(1, -1)
    print('Generating recommendations...')
    # fetch the nearest k neighbors
    distances, indices = model.kneighbors(test_data, n_neighbors=k)
    # top k similar users
    top_k_users_indices = indices.flatten()
    similar_users = matrix.index[top_k_users_indices].tolist()
    # ratings by the similar users
    similar_user_ratings = df[df['userId'].isin(similar_users)]
    # suggest movies based on similar users
    # recommend n top movies 
    recommended_movies = similar_user_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).index[:n]
    end = time.time()
    print("Time taken to predict:", end-start)
    print(f"Top {n} movie recommendations for user:")
    return list(recommended_movies)

def movie_recs(model, user, trainBool=False):
    # train the model if bool is true
    model = train(trainBool, model)
    # return recommendations for a user
    return predict(user, model, k)
    
## CLI testing before model loading was on flask
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("person", type=int)
    parser.add_argument('trainBool', nargs='?', default=False)
    
    args = parser.parse_args()

    person = args.person
    # set to 1 to train model from scratch and then predict
    # deployed ver loads an existing model (default trainBool=false)
    trainBool = args.trainBool

    # print(movie_recs(model, person, trainBool))
    n = 1
    people = df['userId'].unique().tolist()[:n]
    count = 0
    only_good = df.loc[df['rating'] >= 3]
    for i in range(n):
        movies = only_good.loc[only_good['userId'] == people[i], 'movieId']
        movies = movies.to_list()
        genres = []
        for mov in movies:
            temp = df_movies.loc[df_movies['movieId'] == mov, 'genre'].to_list()[0][2:-2].split("\', \'")
            genres += temp
        recs = movie_recs(model, people[i], trainBool)
        for rec in recs:
            rec_gens = df_movies.loc[df_movies['movieId'] == rec, 'genre'].to_list()[0][2:-2].split("\', \'")
            for gen in rec_gens:
                if gen in genres:
                    count += 1
                    break
    print(count/(20*1000))



    