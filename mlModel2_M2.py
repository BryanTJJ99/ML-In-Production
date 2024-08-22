import pandas as pd
import numpy as np


def test(user, model, sparse_matrix, mat_ind, df, top_rated_movies, k = 20):
    # check if user is present in the data a.k.a old user
    # if new user, we have no prior data so we recommend n top rated movies randomly
    if user not in mat_ind:
        #GET INFO FOR HOW TO MAKE THIS WORK FOR NEW USER MAKE SIMILAR FROM TRISHA
        print('New User. Here are some movies rated 5/5!')
        recommended_movies = np.random.choice(top_rated_movies, k).tolist()
        temp = []
        for i in range(k):
            #print(recommended_movies[i])
            temp.append(recommended_movies[i])
        return temp
    
    # convert the pivot table indexing (same as user) to csr matrix indexing
    sp_matrix_user_index = list(mat_ind).index(user)
    # data point to predict
    test_data = sparse_matrix[sp_matrix_user_index].reshape(1, -1)
    print('Generating recommendations...')
    # fetch the nearest k neighbors
    distances, indices = model.kneighbors(test_data, n_neighbors=k)
    # top k similar users
    top_k_users_indices = indices.flatten()
    similar_users = mat_ind[top_k_users_indices].tolist()
    # ratings by the similar users
    similar_user_ratings = df[df['userId'].isin(similar_users)]
    # suggest movies based on similar users
    # recommend n top movies 
    recommended_movies = list(similar_user_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).index[:k])
    temp = []
    for i in range(len(recommended_movies)):
        temp.append(recommended_movies[i])
    print(f"Top {len(recommended_movies)} movie recommendations for user:")
    return temp

def movie_recs(model, user, sparse_matrix, mat_ind, df, top_rated_movies):

    return test(user, model, sparse_matrix, mat_ind, df, top_rated_movies)
    
    