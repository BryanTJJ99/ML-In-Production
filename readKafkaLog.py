import csv
import requests
import json

ratings_lines = []
movies = set()
users = set()
stream_lines = []
with open('kafka_log.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    if row[2].split(" ")[1].startswith("/rate/"):
      title = row[2][len("GET /rate/"):row[2].find("=")]
      user_id = row[1]
      ratings_lines.append([row])
      users.add(user_id)
      movies.add(title)
    else:
      stream_lines.append([row])

movie_lines = []
title_to_id = {}
failed_movies = []
for movie in movies:
    try:
        resp = requests.get(f'http://128.2.204.215:8080/movie/{movie}')
        data = json.loads(resp.content.decode())
        movie_lines.append([data["id"], #id
                                data["tmdb_id"], # tmdb_id
                                data["vote_average"], # vote_avg
                                data["vote_count"], # vote_count
                                data["title"], # title
                                data['original_language'], # language
                                data["adult"], # adult
                                [genre['name'] for genre in data['genres']], # genres
                                data["runtime"], # runtime
                                data["popularity"], # popularity
                                ])
        title_to_id[movie] = data["tmdb_id"]
    except:
        failed_movies.append(movie)

title_to_id_legacy = {}
for movie in movie_lines:
  title_to_id_legacy[movie[0]] = movie[1]

new_rating_lines = []
failed_ratings = []
for row in ratings_lines:
  try:
    row = row[0]
    title = row[2][len("GET /rate/"):row[2].find("=")]
    # datetime, user_id, movie_id, rating,
    new_rating_lines.append([row[0],row[1],
                            title_to_id_legacy[title],
                            title_to_id[title],
                            row[2][row[2].find("=")+1:],
                            ])
  except:
    failed_ratings.append(row)

user_lines = []
failed_users = []
for user in users:
  try:
    user_resp = requests.get(f'http://128.2.204.215:8080/user/{user}')
    user_data = json.loads(user_resp.content.decode())
    user_lines.append([user_data['user_id'], # user_id
                      user_data['age'], # age
                      user_data["occupation"], # occupation
                      user_data['gender'], # gender
    ])
  except:
    failed_users.append(user)

for movie in movie_lines:
  new_title = "".join(movie[4].split("'"))
  movie[4] = new_title



with open('ratings.csv', 'w+') as csvfile:
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerows(new_rating_lines)
with open('users.csv', "w+") as csvfile:
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerows(user_lines)
with open('movies.csv', "w+") as csvfile:
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerows(movie_lines)
with open('streams.csv', "w+") as csvfile:
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerows(stream_lines)

movie_lines = []
with open('movies.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    genres = row[6]
    new_genres = row[6].replace("'", "\"\"")
    new_genres = new_genres.replace("\"[", "\"{")
    new_genres = new_genres.replace("[", "\"{")
    new_genres = new_genres.replace("]\"", "}\"")
    new_genres = new_genres.replace("]", "}\"")
    row[6] = new_genres
    movie_lines.append(row)
    
with open('movies.csv', "w+") as csvfile:
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerows(movie_lines)