import db_utils.config as config
from datetime import datetime
import psycopg2
import requests
import json
from psycopg2.extras import RealDictCursor, execute_values
import re
from collections import defaultdict

def get_connection(database=config.DATABASE):
  conn = psycopg2.connect(database=database,
                        user=config.DB_USER,
                        host=config.HOST,
                        password=config.PASSWORD,
                        port=config.PORT)
  
  return conn

def fetch_movie(conn: psycopg2.extensions.connection, movie_id, commit=True):
  #conn is a connection to the database
  cur = conn.cursor(cursor_factory=RealDictCursor)
  cur.execute("SELECT * FROM movies WHERE id = %s", (movie_id,))
  movie = cur.fetchone()

  if not movie:
    # if this movie id doesn't exist yet, fetch the information and add it to database
    resp = requests.get(f'{config.MOVIE_ENDPOINT}{movie_id}')
    data = json.loads(resp.content.decode())
    movie = {"id": movie_id}
    movie["imdb_id"] = data["imdb_id"] if "imdb_id" in data else "" #imdb_id
    movie["tmdb_id"] = data["tmdb_id"] if "tmdb_id" in data else 0 # tmdb_id
    movie["vote_average"] = data["vote_average"] if "vote_average" in data else 0.0 # vote_avg
    movie["vote_count"] = data["vote_count"] if "vote_count" in data else 0 # vote_count
    movie["title"] = data["title"] if "title" in data else ""# title
    movie["original_language"] = data['original_language'] if "original_language" in data else ""# language
    movie["adult"] = data["adult"] if "adult" in data else False # adult
    movie["genres"] = [genre['name'] for genre in data['genres']] if "genres" in data else []  # genres
    movie["runtime"] = data["runtime"] if "runtime" in data else 0 # runtime
    movie["popularity"] = data["popularity"] if "popularity" in data else 0.0# popularity

    sql = "INSERT INTO movies(id, imdb_id, tmdb_id, vote_avg, vote_count, title, language, adult, genres, runtime, popularity) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    cur.execute(sql, (movie['id'], movie['imdb_id'], movie['tmdb_id'], movie['vote_average'], movie['vote_count'], movie['title'], movie['original_language'],movie['adult'],
                      "{" + ",".join('"' + str(x) + '"' for x in movie['genres']) + "}",movie['runtime'],movie['popularity']))
  elif movie['runtime'] == 0:
    resp = requests.get(f'{config.MOVIE_ENDPOINT}{movie_id}')
    data = json.loads(resp.content.decode())
    update_query = "UPDATE movies SET runtime = %s WHERE id = %s;"
    new_runtime = data["runtime"] if "runtime" in data else -1
    cur.execute(update_query, (new_runtime, movie_id))

  if commit:
    conn.commit()
  cur.close()

  return movie

def fetch_user(conn: psycopg2.extensions.connection, user_id, commit=True):
  cur = conn.cursor(cursor_factory=RealDictCursor)
  cur.execute("SELECT * FROM users WHERE userid = %s", (user_id,))
  user = cur.fetchone()

  if not user:
    resp = requests.get(f'{config.USER_ENDPOINT}{user_id}')
    data = json.loads(resp.content.decode())
    user = {"userid": user_id}
    user["age"] = data["age"] if "age" in data else 0
    user["occupation"] = data["occupation"] if "occupation" in data else ""
    user["gender"] = data["gender"] if "gender" in data else ""

    sql = "INSERT INTO users(userid, age, occupation, gender) VALUES(%s, %s, %s, %s)"
    cur.execute(sql, (user['userid'], user['age'], user['occupation'], user['gender']))
    if commit:
      conn.commit()
  cur.close()

  return user

def parse_get(line):
  '''
    Function for parsing a GET line
  '''
  # Preserve line for error documentation
  parsed_data = {'line': line}
  comma_seperated = line.split(',')
  parsed_data["datetime"] = datetime.strptime(comma_seperated[0], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S')
  parsed_data["userid"] = int(comma_seperated[1])
  movie_info_split = comma_seperated[2].split(' ')[1].split('/')
  kind = "data"
  if movie_info_split[1] == 'data':
    parsed_data["movieid"] = movie_info_split[3]
    movie_minute = int(movie_info_split[4][:-4])
  elif movie_info_split[1] == 'rate':
    kind = "rate"
    parsed_data["movieid"] = movie_info_split[2].split("=")[0]
    parsed_data["rating"] = int(movie_info_split[2].split("=")[1])

  return kind, parsed_data

def parse_recommendation(line):
  '''
    Function for parsing a recommendation line
  '''
  # Preserve line for error documentation
  parsed_data = {'line': line}
  comma_seperated = line.split(',')
  parsed_data["datetime"] = datetime.strptime(comma_seperated[0], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%dT%H:%M:%S')
  parsed_data["userid"] = comma_seperated[1]
  parsed_data["server"] = comma_seperated[2].split(" ")[2]
  parsed_data["status"] = int(comma_seperated[3][1:].split(" ")[1])
  results = []
  results.append(comma_seperated[4][1:].split(" ")[1])
  for i in range(5,len(comma_seperated)-1):
    results.append(comma_seperated[i][1:])
  parsed_data["results"] = results
  parsed_data["responsetime"] = int(comma_seperated[24][1:-3])

  return parsed_data

def parse_recommendation_error(line):
  parsed_data = {'line': line}
  comma_seperated = line.split(',')
  parsed_data["datetime"] = datetime.strptime(comma_seperated[0], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%dT%H:%M:%S')
  parsed_data["userid"] = comma_seperated[1]
  parsed_data["server"] = comma_seperated[2].split(" ")[2]
  parsed_data["status"] = int(comma_seperated[3][1:].split(" ")[1])
  parsed_data["results"] = [comma_seperated[4].strip()]
  parsed_data["responsetime"] = int(comma_seperated[5][1:-3])

  return parsed_data
  
def parse_line(line):
  get_regex = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d{2,},GET\s\/data\/m\/[a-zA-Z0-9+-_*]+\/\d+\.mpg$'
  rate_regex = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2},\d{2,},GET\s\/rate\/[a-zA-Z0-9+-_*]+=[0-5]{1}$'
  reco_regex = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+,\d+,recommendation request [\d\w\-\.]+:\d+, status \d+, result: [a-zA-Z0-9+-_*]+, ([a-zA-Z0-9+-_*]+, ){19}\d+ ms$'
  reco_refuse_regex = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+,\d+,recommendation request [\d\w\-\.]+:\d+, status \d+, result: .+, \d+ ms$'
  if re.match(get_regex, line):
    return parse_get(line)
  elif re.match(rate_regex, line):
    return parse_get(line)
  elif re.match(reco_regex, line):
    return 'recommendation', parse_recommendation(line)
  elif re.match(reco_refuse_regex, line):
    return 'recommendation', parse_recommendation_error(line)
  else:
    return None, None
  
def insert_ratings(conn: psycopg2.extensions.connection, rate_lines):
  cur = conn.cursor()
  cur = conn.cursor()
  ratings_data = [(rating['datetime'], rating['userid'], rating['movieid'], rating['rating']) for rating in rate_lines]
  sql = "INSERT INTO ratings(datetime, userid, movieid, rating) VALUES %s"
  execute_values(cur, sql, ratings_data)

  conn.commit()
  cur.close()

def insert_views(conn: psycopg2.extensions.connection, view_lines):
  cur = conn.cursor()
  
  sql = """INSERT INTO views (datetime, userid, movieid, watchtime)
    VALUES %s
    ON CONFLICT (userid, movieid)
    DO UPDATE SET 
    datetime = excluded.datetime,
    watchtime = views.watchtime + 1;"""

  view_data = defaultdict(lambda: (None, None, None, 0))
  for view in view_lines:
    # Extract relevant information from the current view
    userid = view['userid']
    movieid = view['movieid']
    datetime = view['datetime']
    # Update the defaultdict with the new information
    # Incrementing count and updating other values
    datetime_existing, userid_existing, movieid_existing, count_existing = view_data[(userid, movieid)]
    view_data[(userid, movieid)] = (datetime or datetime_existing, userid or userid_existing, movieid or movieid_existing, count_existing + 1)
  
  execute_values(cur, sql, view_data.values())

  conn.commit()
  cur.close()

def insert_recommendations(conn: psycopg2.extensions.connection, recommendation_lines):
  cur = conn.cursor()
  recommendations_data = [(rec['datetime'], rec['userid'], rec['server'], rec['status'], 
                    "{" + ",".join('"' + str(x) + '"' for x in rec['results']) + "}", rec['responsetime']) for rec in recommendation_lines]
  sql = "INSERT INTO recommendations(datetime, userid, server, status, results, responsetime) VALUES %s"
  execute_values(cur, sql, recommendations_data)

  conn.commit()
  cur.close()

def insert_errors(conn: psycopg2.extensions.connection, error_lines):
  cur = conn.cursor()
  errors_data = [(error["type"], error["line"]) for error in error_lines]
  sql = "INSERT INTO errors(kind, line) VALUES %s"
  execute_values(cur, sql, errors_data)

  conn.commit()
  cur.close()