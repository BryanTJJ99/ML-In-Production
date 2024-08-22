import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.model_selection import cross_val_score
import pickle
import argparse
from scipy.sparse import csr_matrix
import joblib

import pytest
import shutil
from mlModel2 import train, predict, movie_recs


# Logic Attribution: Code inspiration and logic taken from Copilot AI companion, in accordance with course policy

# Data Loading
# Load movie ratings data from a CSV file
df = pd.read_csv('ratings.csv', names = ['date', 'userId', 'movieId', "title", 'rating'])

# Data Preprocessing
# Create a mapping from movie IDs to titles for future reference
df_mapping = df.iloc[:,[2,3]].drop_duplicates()
# Keep only the relevant columns for the recommendation system
df = df.iloc[:,[1,2,4]]
# Create a dictionary for quick lookup of movie titles based on movie ID
title_map = dict(zip(df_mapping["movieId"], df_mapping["title"]))


# Recommendation Parameters
k = 20 # no of recommendations for  existing users
n = 20 # for new users recommend n movies randomly which are rated 5
top_rated_movies = df[df['rating'] == 5]['movieId'].unique()

# model loading for CLI testing - statement not used for deployed version
# loaded in flask for faster prediction 
model = joblib.load('nearest_neigh_model.joblib')

# user-movie matrix
# Pivot the DataFrame to create a matrix with user IDs as rows, movie IDs as columns, and ratings as values
matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
# convert it to sparse format for more efficient storage & computation
sparse_matrix = csr_matrix(matrix.values)
    
    
        
# TEST FUNCTIONS
@pytest.fixture
def load_data():
    """
    Fixture to load the raw data from the CSV file for testing.
    """
    df = pd.read_csv('ratings.csv', names=['date', 'userId', 'movieId', "title", 'rating'])
    return df

def test_data_loading(load_data):
    """
    Test to ensure that the data is loaded correctly and contains the expected columns.
    """
    df = load_data
    assert not df.empty, "Dataframe should not be empty"
    assert set(df.columns) == {'date', 'userId', 'movieId', 'title', 'rating'}, "Dataframe should have the correct columns"

def test_data_preprocessing(load_data):
    """
    Test to ensure that the data preprocessing steps produce the expected outputs.
    """
    df = load_data
    df_mapping = df.iloc[:, [2, 3]].drop_duplicates()
    assert not df_mapping.empty, "Mapping dataframe should not be empty"
    assert 'movieId' in df_mapping.columns and 'title' in df_mapping.columns, "Mapping dataframe should have movieId and title columns"
    
    df_ratings = df.iloc[:, [1, 2, 4]]
    assert 'userId' in df_ratings.columns and 'movieId' in df_ratings.columns and 'rating' in df_ratings.columns, "Ratings dataframe should have userId, movieId, and rating columns"

    
    
    
@pytest.mark.parametrize("train_bool", [True,False])
def test_train_function(train_bool):
    """
    Test to verify that the train function behaves correctly when training is enabled or disabled.
    """
    # Specify the source file path
    source_file_path = 'nearest_neigh_model.joblib'

    # Specify the destination file path (the new copy)
    destination_file_path = 'nearest_neigh_model_backup.joblib'

    # Copy the file
    shutil.copy(source_file_path, destination_file_path)

    # Execute the function with the model and specified bool value
    result_model = train(train_bool, model)

    if train_bool:
        # If training should occur, check that the result is not the original model (new model trained)
        is_trained = result_model is not model
    else:
        # If not training, the original model should be returned unchanged
        is_trained = result_model is model
    
    shutil.copy(destination_file_path, source_file_path)

    assert is_trained == True

def test_new_user_behavior():
    """
    Test to verify the behavior of the system for new users without any previous ratings.
    """
    new_user_id = max(df['userId']) + 1 # A new user ID
    recommended_movies = predict(new_user_id, model, k)
    
    assert len(recommended_movies) == n

def test_movie_recs_for_new_user():
    """
    Test to ensure that new users receive the expected number of movie recommendations.
    """
    new_user_id = max(df['userId']) + 1
    recommendations = movie_recs(model, new_user_id, trainBool=False)
    
    assert len(recommendations) == n, "Expected n recommendations for a new user"
    assert all(movie in title_map.values() for movie in recommendations), "Recommendations should match titles in the title_map"
