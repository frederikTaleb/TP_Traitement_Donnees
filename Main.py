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
# gros biais en faveur de CAN et US par rapport aux autres pays env 60% de l'échantillon
population_pays_chart = profiles['country'].value_counts()
plt.figure(figsize=(12,12))
plt.pie(population_pays_chart, labels = population_pays_chart.index, autopct='%1.1f%%', startangle=140)
plt.title('Répartition des auteurs par région géographique')
plt.show()


# Population ayant déclaré leurs valeurs mbti ET ocean, seulement 377
#pop_mbti_ocean garde le score mbti et le genre pcq je voulais voir s'il pouvait y avoir un lien, pas encore fait
pop_mbti_ocean = profiles[['gender','mbti', 'introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism']].dropna(subset=['mbti', 'introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism'])
mbti_ocean_seul = pop_mbti_ocean[['introverted','intuitive','thinking','perceiving', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism']]
corr_mbti_ocean = mbti_ocean_seul.corr()
#Pour avoir le heatmap
sns.heatmap(corr_mbti_ocean, annot=True, cmap='coolwarm')
plt.show()

#Population avec toutes les valeurs océan
# Problème d'échelle avec les valeurs OCEAN, parfois -1.0 partout
pop_ocean = profiles[['gender', 'agreeableness', 'openness','conscientiousness','extraversion','neuroticism']].dropna(subset=['gender','agreeableness', 'openness','conscientiousness','extraversion','neuroticism'])
pop_ocean_clean = pop_ocean[(pop_ocean['agreeableness'] != -1.0)] # bon, on pourrait s'assurer que toutes les colonnes ocean sont vérifiée...
pop_ocean_clean.groupby('gender').mean()
pop_ocean_clean.drop(columns=['gender']).mean()
# on peut void que les femme sont généralement plus agréables mais aussi plus neurotiques, exploitable?


#Distribution des scores ocean pour chaque type de personnalite mbti
# Je pense qu'on peut tirer une info intéressante pour valider une prédiction
# Serait-il valide de créer un échantillon aléatoire à partir de ces données pour entraîner le modèle de prédiction?
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

# autre piste, faire un binning pour les heures des posts