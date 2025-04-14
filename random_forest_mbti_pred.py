import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from randomForest import randomForest as rf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import text_preprocessing
import load_datasets

mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

df_profiles = load_datasets.load_profiles_dataset()
#df_profiles = df_profiles.dropna(subset=mbti_columns)
df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())

df_comments['created_utc'] = pd.to_datetime(df_comments['created_utc'], unit='s')

aggregation_dict = {
    # Total posts
    'id': 'count',
    # Score
    'score': ['mean', 'median', 'std', 'sum'],
    # Ups
    'ups': ['mean', 'median', 'sum'],
    # Word count
    'word_count': ['mean', 'median', 'std', 'sum'],
    # Controversiality
    'controversiality': ['mean', 'sum'],
    # Gilded 
    'gilded': 'sum',
    # Unique subreddits
    'subreddit_id': pd.Series.nunique, 
    # Timestamps 
    'created_utc': ['min', 'max']
}

author_activity = df_comments.groupby('author').agg(aggregation_dict)
time_span = (author_activity[('created_utc', 'max')] - author_activity[('created_utc', 'min')]) + pd.Timedelta(seconds=1)
# Convert to nb of days
author_activity[('activity_duration_days')] = time_span.dt.total_seconds() / (24 * 60 * 60)
# Calculate posts per day
total_posts = author_activity[('id', 'count')]
duration_days = author_activity[('activity_duration_days')]

min_duration_threshold = 0.01 # Approx 15 mins min activity time
author_activity[('posts_per_day')] = np.where(
    duration_days > min_duration_threshold,
    total_posts / duration_days,
    0 # Don't keep outliers
)

# Flatten
author_activity.columns = ['_'.join(col).strip('_') for col in author_activity.columns.values]

# Rename 
author_activity = author_activity.rename(columns={
    'id_count': 'total_posts',
    'score_mean': 'avg_score',
    'score_median': 'median_score',
    'score_std': 'std_dev_score',
    'score_sum': 'total_score',
    'ups_mean': 'avg_ups',
    'ups_median': 'median_ups',
    'ups_sum': 'total_ups',
    'downs_mean': 'avg_downs',
    'downs_median': 'median_downs',
    'downs_sum': 'total_downs',
    'word_count_mean': 'avg_word_count',
    'word_count_median': 'median_word_count',
    'word_count_std': 'std_dev_word_count',
    'word_count_sum': 'total_word_count',
    'word_count_quoteless_mean': 'avg_word_count_ql',
    'word_count_quoteless_median': 'median_word_count_ql',
    'word_count_quoteless_std': 'std_dev_word_count_ql',
    'word_count_quoteless_sum': 'total_word_count_ql',
    'controversiality_mean': 'controversiality_rate',
    'controversiality_sum': 'controversial_posts_count',
    'gilded_sum': 'total_gilded',
    'subreddit_id_nunique': 'unique_subreddits'
})

# Drop min max time columns
author_activity = author_activity.drop(columns=['created_utc_min', 'created_utc_max'], errors='ignore')
author_activity = author_activity.reset_index()

df_merged = pd.merge(
    left=df_profiles,
    right=author_activity,
    on='author',
    how='left',
    validate='one_to_one' 
)

activity_metric_cols = [
    'total_posts', 'avg_score', 'median_score',
    'avg_ups', 'controversiality_rate', 'total_gilded',
    'avg_word_count', 'median_word_count', 'unique_subreddits',
    'posts_per_day', 'activity_duration_days'
]

other_numeric_cols = ['is_female_pred']

all_relevant_cols = mbti_columns + activity_metric_cols + other_numeric_cols
df_corr = df_merged[all_relevant_cols].copy()
correlation_matrix = df_corr.corr() 

mbti_vs_metrics_corr = correlation_matrix.loc[mbti_columns, [col for col in activity_metric_cols + other_numeric_cols if col in df_corr.columns]]

features = [col for col in activity_metric_cols + ['is_female_pred'] if col in df_merged.columns]
df_merged_train = df_merged.dropna(subset=mbti_columns)
X = df_merged_train[features]

models = {}
for mbti_target in mbti_columns:
    y = df_merged_train[mbti_target].astype(int)
    model = rf(X,y)
    model.train()
    models[mbti_target] = model

for mbti_target in mbti_columns:
    y_pred = models[mbti_target].predict(df_merged[features])
    new_column_name = f"{mbti_target}_pred"
    df_merged[new_column_name] = y_pred

df_merged[['author','introverted_pred', 'intuitive_pred', 'thinking_pred', 'perceiving_pred']].to_csv('datasets/profiles_mbti_pred')