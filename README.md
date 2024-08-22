### Setting Up the Environment

1. **Create and Activate a Virtual Environment**

```
python3 -m venv venv
source venv/bin/activate
```

2. **Installing Required Packages**

```
pip install -r requirements.txt
```

3. **Running Flask**

```
python3 app.py
```
4. **Configure Git with Your Username and Email**

```
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

5. **Configure Database**

First we need to setup the tunnel
```
ssh -L 9092:localhost:9092 tunnel@128.2.204.215 -NT
```
Next we need to setup the postgresql db. 

Install postgresql
```
sudo apt install postgresql postgresql-contrib
```
You can now create a new database and start it.
```
initdb -D database
pg_ctl -D database -l logfile start
```
Create a new user
```
createuser --interactive
```
Now you need to create an inner database and start an interactive session
```
createdb prod
psql prod
```
Now we can create the tables
```
CREATE TABLE ratings(
            datetime TIMESTAMP,
            userid INT,
            movieid TEXT,
            rating INT);

CREATE TABLE movies(
            id TEXT,
            imdb_id TEXT,
            tmdb_id INT,
            vote_avg REAL,
            vote_count INT,
            title TEXT,
            language TEXT,
            adult BOOLEAN,
            genres TEXT[],
            runtime INT,
            popularity FLOAT(6)
            );

CREATE TABLE views(
              datetime TIMESTAMP,
              userid INT,
              movieid TEXT,
              watchtime INT,
              movielength INT);

CREATE TABLE users(
              userid INT,
              age INT,
              occupation TEXT,
              gender CHAR(1));

CREATE TABLE recommendations(
    datetime TIMESTAMP,
    userid INT,
    server TEXT,
    status INT,
    results TEXT[],
    responsetime INT
    );

CREATE TABLE errors(
    kind TEXT,
    line TEXT
    );
```
After creating the tables, rename config_example.py to config.py and fill in the information with your info.
Then you can run the stream and data will start streaming into the database!
```
python db_utils/stream.py
```

### Train Models (not necessary unless new data is added)

Collaborative Filtering:
 ```
python3 mlModel1.py 1 True
```
Nearest Neighbors Model:
```
python3 mlModel2.py 1 1
```
