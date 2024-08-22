import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
from scipy.sparse import csr_matrix
import joblib
import sys
sys.path.append('./db_utils')
import utils
import datetime

def get_info(): # Logic Attribution: Code inspiration and logic taken from Copilot AI companion, in accordance with course policy
    conn = utils.get_connection("current")
    sql = "SELECT userid, movieid, rating FROM ratings LIMIT 10000;"
    sql_query = pd.read_sql_query (sql, conn)
    df = pd.DataFrame(sql_query, columns = ['userid', 'movieid', 'rating'])

    conn = utils.get_connection()
    sql = "SELECT userid, movieid, rating FROM ratings LIMIT " + str(int(10000-len(df))) + ";"
    sql_query = pd.read_sql_query (sql, conn)
    df_bottom = pd.DataFrame(sql_query, columns = ['userid', 'movieid', 'rating'])

    df = pd.concat([df, df_bottom],ignore_index=True)
    df = df.rename(columns={"userid": "userId", "movieid": "movieId", "rating": "rating"})
    # df = pd.read_csv('ratings.csv', names = ['date', 'userId', 'movieId', "title", 'rating'])

    # movie_Ids = pd.DataFrame()
    # temp = df["movieId"].unique()
    # movie_Ids["movie_Ids"] = temp
    # movie_Ids["indices"] = movie_Ids.index
    # title_map = dict(zip(movie_Ids["indices"], temp))

    # for new users recommend n movies randomly which are rated 5 -> OLD USE UNTIL TRISHA FIGURES OUT THING
    top_rated_movies = df[df['rating'] == 5]['movieId'].unique()

    # user-movie matrix
    # if user hasn't rated a movie value is zero

    matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    # convert it to sparse format

    sparse_matrix = csr_matrix(matrix.values)


    # joblib.dump(title_map, 'title_map.joblib')
    joblib.dump(sparse_matrix, 'sparse_matrix.joblib')
    joblib.dump(matrix.index, 'mat_ind.joblib')
    joblib.dump(df, 'df.joblib')
    joblib.dump(top_rated_movies, 'top_rated_movies.joblib')

    print('Training a new model..')
    # cosine similarity
    # nearest neighbor model, n_jobs for parallelization
    model =  NearestNeighbors(metric='cosine',  algorithm = "brute", n_jobs=-1)
    model.fit(sparse_matrix)
    # save model to the file, replaces existing model!
    joblib.dump(model, 'nearest_neigh_model.joblib')
    print('Model training & saving complete!')

prev_time = datetime.datetime(2024, 8, 25, 14, 8, 56)
while True:
    current_time = datetime.datetime.now()
    if not (datetime.timedelta(seconds=-3600) <= (current_time - prev_time) <= datetime.timedelta(seconds=3600)):
        get_info()

        