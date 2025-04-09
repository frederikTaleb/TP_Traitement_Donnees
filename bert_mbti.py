import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import text_preprocessing

from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW

mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

# Isolation des auteurs ayant des valeurs MBTI et leurs commentaires en anglais - nécessaire pour l'entraînement
df_profiles = load_datasets.load_profiles_dataset()
df_profiles = df_profiles.dropna(subset=mbti_columns)
df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())
df_comments = df_comments[df_comments['lang'] == 'en']

# Retrait du HTML et de liens URL vu qu'ils ne sont pas interprétables par BERT
df_comments = df_comments['body'].apply(text_preprocessing.remove_html_tags)
df_comments = df_comments['body'].apply(text_preprocessing.remove_urls)

# compte des commentaires par auteur
comment_counts = df_comments['author'].value_counts()
plt.figure(figsize=(10, 6))
comment_counts.hist(bins = len(df_profiles))
plt.title('Nombre de commentaires par auteur')
plt.ylabel('Nombre d\'auteurs')
plt.xlabel('Nombre de commentaires')
plt.xlim(0,5000)
plt.tight_layout()
# On assume que les auteurs n'ayant pas assez de commentaires ne pourront pas remplire un corpus intéressant
# hyperparamètre : nb_commentaire_min - 20



# compte du nombre de mots par commentaire
word_counts = df_comments['word_count']
plt.figure(figsize=(10,6))
word_counts.hist(bins = len(df_comments['word_count'].unique()))
plt.title('Nombre de mots par post')
plt.ylabel('Nombre de mots')
plt.xlabel('Nombre de posts')
plt.xlim(0,200)
plt.tight_layout()
# On assume que les commentaires ayant trop peu de mots ne contriueront pas à la classification
# hyperparamètre : nb_mots_min (dans un commentaire) - 5

# compte # autheur / mbti
mbti_counts = df_profiles['mbti'].value_counts()
plt.figure(figsize=(10, 6))
mbti_counts.plot(kind='bar', color='gray')
plt.title('Nombre d\'auteurs par type MBTI')
plt.xlabel('Type MBTI')
plt.ylabel('Nombre d\'auteur')
plt.tight_layout()
# Échantilonnage aléatoire stratifié par type mbti