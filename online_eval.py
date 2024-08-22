import db_utils.utils as utils
import pandas as pd
import matplotlib.pyplot as plt
from psycopg2.extras import RealDictCursor

# Connect to the database with older data
conn = utils.get_connection()

# Connect to the database with current data
#conn = utils.get_connection(database = "current")

# Fetch recommendations data
cur = conn.cursor(cursor_factory=RealDictCursor)
cur.execute("SELECT * FROM recommendations;")
recommendations = cur.fetchall()
df_recommendations = pd.DataFrame(recommendations)

# Fetch views data
cur.execute("SELECT * FROM views;")
views = cur.fetchall()
df_views = pd.DataFrame(views)

# Fetch ratings data
cur.execute("SELECT * FROM ratings;")
ratings = cur.fetchall()
df_ratings = pd.DataFrame(ratings)

# Close the connection
conn.close()

# Ensure datetime columns are in the correct format
df_ratings['datetime'] = pd.to_datetime(df_ratings['datetime'])
df_views['datetime'] = pd.to_datetime(df_views['datetime'])
df_recommendations['datetime'] = pd.to_datetime(df_recommendations['datetime'])

recommendations_expanded = df_recommendations[['datetime', 'userid', 'results']].explode('results')
recommendations_expanded.rename(columns={'datetime': 'rec_datetime', 'results': 'movieid'}, inplace=True)

# User Satisfaction Calculation
# Merge recommendations and ratings
merged_ratings = pd.merge(recommendations_expanded, df_ratings, on=['userid', 'movieid'], how='left')
merged_ratings = merged_ratings[merged_ratings['datetime'] > merged_ratings['rec_datetime']]
satisfaction_df = merged_ratings[merged_ratings['rating'] >= 4]
user_satisfaction_rate = len(satisfaction_df) / len(merged_ratings)

# Define a time window for considering a "hit" (e.g., 30 days)
time_window = pd.Timedelta(days=30)

# Hit Rate Calculation
# Merge recommendations and views
merged_views = pd.merge(recommendations_expanded, df_views, on=['userid', 'movieid'], how='left')
merged_views = merged_views[merged_views['datetime'] > merged_views['rec_datetime']]
merged_views['time_diff'] = merged_views['datetime'] - merged_views['rec_datetime']
hit_df = merged_views[merged_views['time_diff'] <= time_window]
hit_rate = len(hit_df) / len(recommendations_expanded)

# Plotting Graphs
# User Satisfaction Rate Over Time
satisfaction_rate_daily = merged_ratings.groupby(merged_ratings['datetime'].dt.date)['rating'].apply(lambda x: (x >= 4).mean())
satisfaction_rate_daily.plot(kind='line', figsize=(10, 6))
plt.title('User Satisfaction Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Satisfaction Rate')
plt.show()
plt.savefig("./online_eval_figures/satisfaction_rate_over_time.png")

# Hit Rate Over Time
hit_rate_daily = merged_views.groupby(merged_views['datetime'].dt.date)['time_diff'].apply(lambda x: (x <= time_window).mean())
hit_rate_daily.plot(kind='line', figsize=(10, 6))
plt.title('Hit Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Hit Rate')
plt.show()
plt.savefig("./online_eval_figures/hit_rate_over_time.png")
