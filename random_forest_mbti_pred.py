import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import load_datasets
from randomForest import randomForest as rf

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']
TARGET_TIMEZONE = 'America/Chicago'
tz_info = ZoneInfo(TARGET_TIMEZONE)

df_profiles = load_datasets.load_profiles_dataset()
#df_profiles = df_profiles.dropna(subset=mbti_columns)
df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())

df_comments['created_utc'] = pd.to_datetime(df_comments['created_utc'], unit='s')
df_comments['created_utc_aware']=df_comments['created_utc'].dt.tz_localize('UTC')
df_comments['created_local'] = df_comments['created_utc_aware'].dt.tz_convert(tz_info)
df_comments['local_hour'] = df_comments['created_local'].dt.hour
df_comments['time_of_day'] = df_comments['local_hour'].apply(get_time_of_day)

# Count comments per author per time_of_day category
tod_counts = df_comments.groupby(['author', 'time_of_day']).size()

# Convert counts to a table: authors as rows, time_of_day as columns
tod_counts_table = tod_counts.unstack(fill_value=0)

# Ensure all 4 time categories exist as columns, adding missing ones with 0 counts
all_tod_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
tod_counts_table = tod_counts_table.reindex(columns=all_tod_categories, fill_value=0)

# Calculate proportions by dividing each row by its sum
tod_proportions = tod_counts_table.apply(lambda x: x / x.sum(), axis=1)

# Rename columns for clarity
tod_proportions = tod_proportions.rename(columns={
    'Morning': 'tod_prop_morning',
    'Afternoon': 'tod_prop_afternoon',
    'Evening': 'tod_prop_evening',
    'Night': 'tod_prop_night'
})

df_profiles = pd.merge(
        df_profiles,
        tod_proportions,
        on='author',
        how='left'
)

proportion_cols = ['tod_prop_morning', 'tod_prop_afternoon', 'tod_prop_evening', 'tod_prop_night']
df_proportions = df_profiles[proportion_cols]
max_tod_col_names = df_proportions.idxmax(axis=1)
most_frequent_tod_category = max_tod_col_names.str.replace('tod_prop_', '', regex=False)
new_feature_name = 'most_frequent_tod'
df_profiles[new_feature_name] = most_frequent_tod_category


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

other_numeric_cols = ['is_female_pred','tod_prop_morning', 'tod_prop_afternoon', 'tod_prop_evening', 'tod_prop_night']

all_relevant_cols = mbti_columns + activity_metric_cols + other_numeric_cols

# Matrice de corrélation avec chaque score MBTI, visualisation par heatmap 
df_corr = df_merged[all_relevant_cols].copy()
correlation_matrix = df_corr.corr() 
mbti_vs_metrics_corr = correlation_matrix.loc[mbti_columns, [col for col in activity_metric_cols + other_numeric_cols if col in df_corr.columns]]
print("Correlations between MBTI dimensions and other metrics:")
print(mbti_vs_metrics_corr)
# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    mbti_vs_metrics_corr,
    annot=True,       
    cmap='coolwarm',  
    center=0,         
    fmt=".2f",        
    linewidths=.5     
)
plt.title('Correlation entre les dimensions MBTI et les attributs d\'activité ajoutés')
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)
plt.tight_layout() 
plt.show()

features = [col for col in activity_metric_cols + ['is_female_pred','tod_prop_morning', 'tod_prop_afternoon', 'tod_prop_evening', 'tod_prop_night'] if col in df_merged.columns]
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

for mbti_target in mbti_columns:
    models[mbti_target].getImportance()

for mbti_target in mbti_columns:
    print(mbti_target)
    print(models[mbti_target].accuracy)
    #print(models[mbti_target].cm)

#df_merged[['author','introverted_pred', 'intuitive_pred', 'thinking_pred', 'perceiving_pred']].to_csv('datasets/profiles_mbti_pred')