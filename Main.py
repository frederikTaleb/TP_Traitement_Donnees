import numpy as np
import pandas as pd

import load_datasets


profiles = load_datasets.load_profiles_dataset()
comments = load_datasets.load_comments_dataset()

# Population

# nb d'individus
population_totale = len(profiles)
# nb d'individus avec un genre déclaré
population_genre_notNull = profiles['gender'].notnull().sum()

# nb d'individus par genre avec déclaration
population_genre = profiles.groupby('gender')['gender'].agg(['count'])
# Age moyen, age moyen par genre
ages_tous = profiles[['age']].agg(['mean','std', 'min', 'max'])
ages_par_gender = profiles.groupby('gender')['age'].agg(['mean', 'std','min', 'max','count'])

# résumé population générale

# résumé population genre & age
age_gender_stats = pd.concat([population_genre, ages_par_gender], axis=1)
#age_gender_stats.rename(columns={"count":"cnt_genre","mean":"moyenne_age", "std": "ecart_type_age", "min":"min_age", "max":"max_age"}, inplace=True)
age_gender_stats.colums.values[0] = "cnt_genre"
age_gender_stats.colums.values[0] = "moyenne_age"
age_gender_stats.colums.values[0] = "ecart_type_age"
age_gender_stats.colums.values[0] = "min_age"
age_gender_stats.colums.values[0] = "max_age"
age_gender_stats.colums.values[0] = "cnt_age_declare"

# Répartition des auteurs par région géographique




