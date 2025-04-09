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

import load_datasets

#################################################

class BertForMultiRegression(nn.Module):
    def __init__(self, bert_model, num_outputs):
        super(BertForMultiRegression, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(bert_model.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.linear(pooled_output)

#################################################

class PersonalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

################################################

def ocean_predict():
    df_profiles = load_datasets.load_profiles_ocean_scores()
    df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())

    ocean_columns = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
    mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

    # normalisation des scores OCEAN entre 0 et 1
    for column in ocean_columns:
        df_profiles[column] = df_profiles[column]  / 100

    # transformation de created_utc en datetime
    df_comments['created_datetime'] = pd.to_datetime(df_comments['created_utc'], unit='s')

    # texte concaténé par auteur
    df_comments['body'] = df_comments['body'].astype(str)
    df_author_comments = df_comments.groupby('author')['body'].apply(lambda x: ' '.join(x)).reset_index()
    df_final = pd.merge(df_profiles, df_author_comments, on='author', how='left')

    texts = df_final['body'].tolist()
    labels = df_final[['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']].values.tolist()

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    max_length = 512
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    num_ocean_traits = 5
    model = BertForMultiRegression(bert_model, num_ocean_traits)

    dataset = PersonalityDataset(encodings, labels)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()
    num_epochs = 3  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}")

        # Optional: Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")

def mbti_predict():

    mbti_columns = ['introverted', 'intuitive', 'thinking','perceiving']

    df_profiles = load_datasets.load_profiles_dataset()
    df_profiles = df_profiles.dropna(subset=mbti_columns).all()
    #df_comments = load_datasets.load_comments_by_authors(df_profiles['author'].unique())