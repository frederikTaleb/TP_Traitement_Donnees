import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import load_datasets


profiles = load_datasets.load_profiles_dataset()
comments = load_datasets.load_comments_dataset()

# Étude des données des profiles

# Population

# nb d'individus
population_totale = len(profiles)
# nb d'individus avec un genre déclaré 3227
population_genre_notNull = profiles['gender'].notnull().sum()

# nb d'individus par genre avec déclaration
population_genre = profiles.groupby('gender')['gender'].agg(['count'])
# Age moyen, age moyen par genre
ages_tous = profiles[['age']].agg(['mean','std', 'min', 'max'])
ages_par_gender = profiles.groupby('gender')['age'].agg(['mean', 'std','min', 'max','count'])

# résumé population genre & age
age_gender_stats = pd.concat([population_genre, ages_par_gender], axis=1)
#age_gender_stats.rename(columns={"count":"cnt_genre","mean":"moyenne_age", "std": "ecart_type_age", "min":"min_age", "max":"max_age"}, inplace=True)
age_gender_stats.columns.values[0] = "cnt_genre"
age_gender_stats.columns.values[0] = "moyenne_age"
age_gender_stats.columns.values[0] = "ecart_type_age"
age_gender_stats.columns.values[0] = "min_age"
age_gender_stats.columns.values[0] = "max_age"
age_gender_stats.columns.values[0] = "cnt_age_declare"

# Répartition des auteurs par région géographique
nb_pays_notNull = profiles['country'].notnull().sum() # 2146
population_pays = profiles.groupby('country')['country'].agg(['count'])
# gros biais en faveur de l'amérique du nord par rapport aux autres pays

# Population ayant déclaré leurs valeurs mbti ET ocean, seulement 377
#pop_mbti_ocean garde le score mbti et le genre pcq je voulais voir s'il pouvait y avoir un lien, pas encore fait
pop_mbti_ocean = profiles[['gender','mbti', 'introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism']].dropna(subset=['mbti', 'introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism'])
mbti_ocean_seul = pop_mbti_ocean[['introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism']]
corr_mbti_ocean = mbti_ocean_seul.corr()
#Pour avoir le heatmap

sns.heatmap(corr_mbti_ocean, annot=True, cmap='coolwarm')
plt.show()

#Distribution des scores ocean pour chaque type de personnalite mbti
# Je pense qu'on peut tirer une info intéressante pour valider une prédiction
pop_mbti_ocean_melt = pop_mbti_ocean.melt(id_vars=['mbti'], value_vars=['openness','agreeableness','conscientiousness','extraversion','neuroticism'], var_name='OCEAN', value_name='Score')
plt.figure(figsize=(48,12))
sns.boxplot(x='mbti', y='Score', hue='OCEAN', data=pop_mbti_ocean_melt)
plt.title('Dist des scores ocean pour chaque type de personnalite mbti')
plt.xlabel('MBTI pers.')
plt.ylabel('Ocean Score')
plt.legend(title='Attributs OCEAN')
plt.show()

#Étude des données pour les commentaires
comments_author = comments['author'].value_counts()
nb_comment_moyen = comments_author.mean()
nb_comment_median = comments_author.median()
nb_comment_ecart_type = comments_author.std()
# la médiane et l'écart type comparés à la moyenne montrent que quelques utilisateurs font des tonnes de commentaires, d'autres très peu

# On pourrait essayer de voir un lien entre le nombre de commentaires et des caractéristiques OCEAN ou MBTI en ajoutant la donnée aux profiles


# Étude des données combinées (à voir le merge semble faire de quoi de bizarre, peut-être inutile)
# profiles_comments = pd.merge(profiles, comments, on='author') # très long env 6 minutes
# je voulais avoir les commentaires avec les auteurs et leurs scores
# profiles_comments_mbti_ocean = profiles_comments.dropna(subset=['mbti', 'introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism'])
