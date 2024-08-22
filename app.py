from flask import Flask
from mlModel2_M2 import movie_recs
import joblib
import datetime

app = Flask(__name__)

model2 = joblib.load('nearest_neigh_model.joblib')
sparse_matrix = joblib.load('sparse_matrix.joblib')
matrix = joblib.load('mat_ind.joblib')
df = joblib.load('df.joblib')
top_rated_movies = joblib.load('top_rated_movies.joblib')


prev_time = datetime.datetime(2024, 8, 25, 14, 8, 56)
def get_recommendations(user_id):
    """Fetch recommendations for a given user ID using the loaded ML model."""
    current_time = datetime.datetime.now()
    #if the current time is more than an hour from the last model update, update the model
    global prev_time
    global model2
    global sparse_matrix
    global matrix
    global df
    global top_rated_movies 

    if not (datetime.timedelta(seconds=-3600) <= (current_time - prev_time) <= datetime.timedelta(seconds=3600)):
        print("Updating model!")
        prev_time = current_time
        model2 = joblib.load('nearest_neigh_model.joblib')
        sparse_matrix = joblib.load('sparse_matrix.joblib')
        matrix = joblib.load('mat_ind.joblib')
        df = joblib.load('df.joblib')
        top_rated_movies = joblib.load('top_rated_movies.joblib')

    user_id = int(user_id)
    
    recommendations = movie_recs(model2, int(user_id), sparse_matrix, matrix, df, top_rated_movies) # model 2
    
    return ",".join(str(item) for item in recommendations)

@app.route("/recommend/<userid>")
def recommend(userid):
    """Endpoint to get movie recommendations for a user."""
    recommendations = get_recommendations(userid)
    return recommendations


if __name__ == "__main__":
    #threaded = True should make every new request in a new thread so if one causes the model update to trigger, the other requests won't get backed up
    app.run(host="0.0.0.0", port=8082, debug=True, threaded = True) 
