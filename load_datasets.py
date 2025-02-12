import pandas as pd

def load_profiles_dataset():
    """
    Charge les profiles du fichier author_profiles_7027.csv

    Retour :
        la fonction retourne un dataframe pandas 
    """

    return pd.read_csv("datasets/author_profiles_7027.csv")

def load_comments_dataset():
    """
    Charge les commentaires du fichier all_comments_since_2015.csv
    """

    return pd.read_csv("datasets/all_comments_since_2015.csv")