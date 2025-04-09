import pandas as pd

def discretize(df, columns, nb_bins):
    """
    À partir d'un dataframe retourne un dataframe des colonnes spécifées discrétisées en nb_bins intervals de largeur égale

    df : un dataframe pandas
    columns : une liste des noms de colonnes à discrétiser
    nb_bins : le nombre d'intervals à créer
    """
    df_discretized = pd.DataFrame()
    for col in columns:
        df_discretized[f'{col}_binned'] = pd.cut(df[col], bins=nb_bins, labels=False, duplicates='drop')
    
    return df_discretized