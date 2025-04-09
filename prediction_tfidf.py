import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import load_datasets
import feature_eng
import text_preprocessing


df_profiles = load_datasets.load_profiles_ocean_scores()
df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())

ocean_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

# normalisation des scores OCEAN entre 0 et 1
for column in ocean_columns:
    df_profiles[column] = df_profiles[column]  / 100

# transformation de created_utc en datetime
df_comments['created_datetime'] = pd.to_datetime(df_comments['created_utc'], unit='s')

# prétraitement du texte concaténé par auteur
#df_comments['body'] = df_comments['body'].astype(str)
#df_author_comments = df_comments.groupby('author')['body'].apply(lambda x: ' '.join(x)).reset_index()
#df_author_comments['body'] = df_author_comments['body'].apply(text_preprocessing.remove_urls)
#df_author_comments['body'] = df_author_comments['body'].apply(text_preprocessing.remove_html)
#df_author_comments['body'] = df_author_comments['body'].apply(text_preprocessing.lower_number_punctuation)
#df_author_comments['body'] = df_author_comments['body'].apply(text_preprocessing.remove_stopwords) #462min 3.4s
#df_author_comments['body'] = df_author_comments['body'].apply(text_preprocessing.remove_whitespace)
# sauvegardé dans author_body_no_stop.csv

df_author_comments = pd.read_csv("datasets/author_body_no_stops.csv")

# Génération des valeur TF-IDF
corpus = df_author_comments['body'].tolist()
authors = df_author_comments['author'].tolist()

max_features = 500
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=authors, columns=tfidf_vectorizer.get_feature_names_out())
df_tfidf.index.name = 'author'
df_tfidf = df_tfidf.reset_index()

df_author_tfidf = pd.merge(df_author_comments, df_tfidf, on='author', how='left')
df_author_merged = pd.merge(df_profiles[['author', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']], df_tfidf, on='author', how='left')

# Génération des valeurs pour la proportion de réponses

comments = df_comments.groupby('author').size()
replies = df_comments[df_comments['parent_id'].str.startswith('t1_', na=False)].groupby('author').size()
proportion_replies = (replies / comments).fillna(0)
df_proportion_replies = proportion_replies.reset_index()
df_proportion_replies.columns = ['author', 'proportion_of_replies']

df_author_merged = pd.merge(df_author_merged, df_proportion_replies, on='author', how='left').fillna(0)

# Génération des valeurs pour l'activité moyenne par jour

activity = df_comments.groupby('author')['created_datetime'].agg(['min', 'max'])
activity.rename(columns={'min': 'first_comment_time', 'max': 'last_comment_time'}, inplace=True)

# nb de jours d'activite
activity['activity_duration'] = (activity['last_comment_time'] - activity['first_comment_time']).dt.days
activity['activity_duration'] = activity['activity_duration'].replace(0, 1)

# nb moyen de posts par jour
comments = comments.rename('total_comments')
activity = activity.merge(comments, on='author')
activity['avg_posts_per_day'] = activity['total_comments'] / activity['activity_duration']
min_val = activity['avg_posts_per_day'].min()
max_val = activity['avg_posts_per_day'].max()
activity['avg_posts_per_day_normalized'] = (activity['avg_posts_per_day'] - min_val) / (max_val - min_val)
activity.reset_index(inplace=True)

# ajout au dataframe
df_avg_activity = activity[['author', 'avg_posts_per_day_normalized']]
df_author_merged = pd.merge(df_author_merged, df_avg_activity, on='author', how='left').fillna(0)


# test avec random forest regression
features = [column for column in df_author_merged.columns if column not in ['author', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]

X = df_author_merged[features]
y = df_author_merged[ocean_columns[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr = SVR()
svr.fit(X_train, y_train)
#random_forest = RandomForestRegressor(random_state=42)
#random_forest.fit(X_train, y_train)

y_pred = svr.predict(X_test)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)