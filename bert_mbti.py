import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

import text_preprocessing
import load_datasets
import MBTIDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) >= 0.5).int().flatten().tolist()
    labels = labels.flatten().astype(int).tolist()
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def predict_mbti_score(text, trained_model, tokenizer, max_length):
    """Prédiction score mbti pour un text"""
    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(**inputs)
        logits = outputs.logits
        prediction_probability = torch.sigmoid(logits).item()
        prediction = 1 if prediction_probability >= 0.5 else 0
    return prediction, prediction_probability

mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

# Isolation des auteurs ayant des valeurs MBTI et leurs commentaires en anglais - nécessaire pour l'entraînement
df_profiles = load_datasets.load_profiles_dataset()
df_profiles = df_profiles.dropna(subset=mbti_columns)
df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())
df_comments = df_comments[df_comments['lang'] == 'en']

# Retrait du HTML et de liens URL vu qu'ils ne sont pas interprétables par BERT
df_comments['body'] = df_comments['body'].apply(text_preprocessing.remove_html_tags)
df_comments['body'] = df_comments['body'].apply(text_preprocessing.remove_urls)

# compte du nombre de mots par commentaire : loi de puissance
word_counts = df_comments['word_count']
plt.figure(figsize=(10,6))
word_counts.hist(bins = len(df_comments['word_count'].unique()))
plt.title('Nombre de mots par post')
plt.ylabel('Nombre de mots')
plt.xlabel('Nombre de posts')
plt.xlim(0,200)
plt.tight_layout()
# On assume que les commentaires ayant trop peu de mots ne contriueront pas à la classification
# échantillonage 
#   hyperparamètre : nb_mots_min (dans un commentaire) - 10
df_comments = df_comments[df_comments['word_count'] > 10]

# compte des commentaires par auteur : loi de puissance
comment_counts = df_comments['author'].value_counts()
plt.figure(figsize=(10, 6))
comment_counts.hist(bins = len(df_profiles))
plt.title('Nombre de commentaires par auteur')
plt.ylabel('Nombre d\'auteurs')
plt.xlabel('Nombre de commentaires')
plt.xlim(0,5000)
plt.tight_layout()
# On assume que les auteurs n'ayant pas assez de commentaires ne pourront pas remplire un corpus intéressant
# échantillonage
#   hyperparamètre : nb_commentaire_min - 20
keep = comment_counts[comment_counts >= 20].index
df_comments = df_comments[df_comments['author'].isin(keep)]

# Plutôt que de se fier aux auteurs, on évalue le score mbti de chaque commentaire
# pour ensuite assigner la majorité à l'auteur
df_author_mbti = df_profiles[['author','introverted', 'intuitive', 'thinking','perceiving']]
df_comments_merged = pd.merge(df_comments, df_author_mbti, on='author', how='left')


# Modèle et entraînement (test pour introversion)
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
num_labels = 1  # Classification binaire
max_length = 256 # longeur max de la liste de tokens, BERT prend max 512
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

df_1 = df_comments_merged[df_comments_merged['introverted'] == 1]
df_0 = df_comments_merged[df_comments_merged['introverted'] == 0]
# temporarire pour test
"""
df_1 = df_1.sample(n=1000, replace=False, random_state=42)
df_0 = df_0.sample(n=1000, replace=False, random_state=42)
df = pd.concat([df_1,df_0])
df = df.dropna(subset=['body', 'introverted'])
"""

df = df_comments_merged.dropna(subset=['body', 'introverted'])
df_1 = df_1.sample(n=2000000, replace=False, random_state=42)
df_0 = df_0.sample(n=2000000, replace=False, random_state=42)
df = pd.concat([df_1,df_0])
texts = df['body'].tolist()
labels = df['introverted'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
train_dataset = MBTIDataset(train_encodings, train_labels)
val_dataset = MBTIDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir='./results_introverted',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs_introverted',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()

evaluation_results = trainer.evaluate()
print(f"Evaluation Results for Introverted: {evaluation_results}")

best_model_path = training_args.output_dir + "/checkpoint-" + str(trainer.state.best_model_checkpoint).split('-')[-1]
best_model = BertForSequenceClassification.from_pretrained(best_model_path)
# Example Prediction
example_text = "I really enjoy spending time with large groups of people and feel energized afterwards."
predicted_score, probability = predict_mbti_score(example_text, best_model, tokenizer, max_length)
print(f"\nExample Prediction for text: '{example_text}'")
print(f"Predicted Introverted Score (1=Introverted, 0=Extraverted): {predicted_score}")
print(f"Probability of being Introverted: {probability:.4f}")

"""
# compte # autheur / mbti
mbti_counts = df_profiles['mbti'].value_counts()
plt.figure(figsize=(10, 6))
mbti_counts.plot(kind='bar', color='gray')
plt.title('Nombre d\'auteurs par type MBTI')
plt.xlabel('Type MBTI')
plt.ylabel('Nombre d\'auteur')
plt.tight_layout()
# Échantilonnage aléatoire stratifié par type mbti
"""