import pandas as pd

def load_profiles_dataset():
    """
    Charge les profiles du fichier author_profiles_7027.csv
    """

    return pd.read_csv("datasets/author_profiles_7027.csv")

def load_comments_dataset():
    """
    Charge les commentaires du fichier all_comments_since_2015.csv
    """

    return pd.read_csv("datasets/all_comments_since_2015.csv")

def load_profiles_mbti_ocean_all_scores():
    """
    Charge les profiles qui ont tous des scores mbti & ocean. 
    Les scores du big 5 qui sont à -1 sont élimés
    """
    df = load_profiles_dataset()
    df = df.dropna(subset=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
    df = df.loc[(df['openness'] != -1.0) & (df['conscientiousness'] != -1.0) & (df['extraversion'] != -1.0) &(df['agreeableness'] != -1.0) &(df['neuroticism'] != -1.0)]
    df = df.dropna(subset=['introverted', 'intuitive', 'thinking','perceiving'])
    return df

def load_profiles_ocean_scores():
    """
    Charge les profiles qui ont tous des scores ocean
    """
    df = load_profiles_dataset()
    df = df.dropna(subset=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
    df = df.loc[(df['openness'] != -1.0) & (df['conscientiousness'] != -1.0) & (df['extraversion'] != -1.0) &(df['agreeableness'] != -1.0) &(df['neuroticism'] != -1.0)]
    return df

def load_comments_by_authors(authors):
    """
    Charge les commentaires seulement pour les auteurs spécifiés

    author : liste des auteurs pour lesquels on charge les commentaires
    """
    df = load_comments_dataset()
    df = df[df['author'].isin(authors)]
    return df